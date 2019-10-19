/* camera.cu - Copyright 2019 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include "noerrors.h"

#define INSTANCEIDX (prim >> 20)
#define HIT_U hitData.x
#define HIT_V hitData.y
#define HIT_T hitData.w
#define RAY_O pos

//  +-----------------------------------------------------------------------------+
//  |  extendPathKernel                                                      |
//  |  extend eye path or light path.                                  LH2'19|
//  +-----------------------------------------------------------------------------+
__global__  __launch_bounds__( 256 , 1 )
void extendEyePathKernel(int smcount, BiPathState* pathStateData,
    Ray4* visibilityRays, Ray4* randomWalkRays, const uint R0, const uint* blueNoise,
    const float spreadAngle, const int4 screenParams, const int probePixelIdx,
    uint* eyePathBuffer, uint* contributionBuffer_Emissive, uint* contributionBuffer_Explicit,
    uint* contributionBuffer_Connection,
    float4* accumulatorOnePass, float NKK)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= counters->extendEyePath) return;

    int jobIndex = eyePathBuffer[gid];

    uint path_s_t_type_pass = pathStateData[jobIndex].pathInfo.w;

    uint pass, type, t, s;
    getPathInfo(path_s_t_type_pass, pass, s, t, type);

    const int scrhsize = screenParams.x & 0xffff;
    const int scrvsize = screenParams.x >> 16;
    const uint x = jobIndex % scrhsize;
    uint y = jobIndex / scrhsize;
    const uint sampleIndex = (pass-1)*12 +(t - 1) * 3 + s;
    y %= scrvsize;

    float3 pos, dir;
    float4 hitData;

    float d, pdf_area, pdf_solidangle;
    float3 throughput, beta;

    throughput = make_float3(pathStateData[jobIndex].data4);
    beta = make_float3(pathStateData[jobIndex].data5);
    pos = make_float3(pathStateData[jobIndex].data6);
    dir = make_float3(pathStateData[jobIndex].data7);

    d = pathStateData[jobIndex].data4.w;
    pdf_area = pathStateData[jobIndex].data5.w;
    pdf_solidangle = pathStateData[jobIndex].data6.w;

    hitData = pathStateData[jobIndex].eye_intersection;

    const int prim = __float_as_int(hitData.z);
    const int primIdx = prim == -1 ? prim : (prim & 0xfffff);

    const CoreTri4* instanceTriangles = (const CoreTri4*)instanceDescriptors[INSTANCEIDX].triangles;

    ShadingData shadingData;
    float3 N, iN, fN, T;
    const float3 I = RAY_O + HIT_T * dir;
    const float coneWidth = spreadAngle * HIT_T;
    GetShadingData(dir, HIT_U, HIT_V, coneWidth, instanceTriangles[primIdx], INSTANCEIDX, shadingData, N, iN, fN, T);

    throughput = beta;
    pdf_area = pdf_solidangle * fabs(dot(-dir, fN)) / (HIT_T * HIT_T);

    float3 R;
    float r4, r5;

    if (false && sampleIndex < 256)
    {
        r4 = blueNoiseSampler(blueNoise, x, y, sampleIndex, 0);
        r5 = blueNoiseSampler(blueNoise, x, y, sampleIndex, 1);
    }
    else
    {
        uint seed = WangHash(jobIndex + R0);
        r4 = RandomFloat(seed);
        r5 = RandomFloat(seed);
    }
    const float3 bsdf = SampleBSDF(shadingData, fN, N, T, dir * -1.0f, r4, r5, R, pdf_solidangle, type);

    /*
    if (jobIndex == probePixelIdx && s == 1)
    {
        printf("roughness:%f\n", ROUGHNESS);
    }
    */

    beta *= bsdf * fabs(dot(fN, R)) / pdf_solidangle;

    const uint randomWalkRayIdx = atomicAdd(&counters->randomWalkRays, 1);
    randomWalkRays[randomWalkRayIdx].O4 = make_float4(SafeOrigin(I, R, N, geometryEpsilon), 0);
    randomWalkRays[randomWalkRayIdx].D4 = make_float4(R, 1e34f);

    s++;

    // the ray is from eye to the pixel directly
    if (jobIndex == probePixelIdx && s == 1)
        counters->probedInstid = INSTANCEIDX,	// record instace id at the selected pixel
        counters->probedTriid = primIdx,		// record primitive id at the selected pixel
        counters->probedDist = HIT_T;			// record primary ray hit distance

    float dE = 1.0f / pdf_area; // N0k
    if (s > 1)
    {
        float3 light_pos = make_float3(pathStateData[jobIndex].data2);
        float3 light2eye = normalize(light_pos - I);

        float bsdfPdf;
        const float3 sampledBSDF = EvaluateBSDF(shadingData, fN, T, light2eye, dir * -1.0f, bsdfPdf);

        float3 normal = make_float3(pathStateData[jobIndex].eye_normal);
        float light_p = bsdfPdf * fabs(dot(normal, dir)) / (HIT_T * HIT_T);


        //dE = (1.0f / pdf_area + d) ;

        dE = (1.0f + light_p * d) / pdf_area;

    }

    pathStateData[jobIndex].data4 = make_float4(throughput, dE);
    pathStateData[jobIndex].data5 = make_float4(beta, pdf_area);;
    pathStateData[jobIndex].data6 = make_float4(I, pdf_solidangle);
    pathStateData[jobIndex].data7 = make_float4(R, __int_as_float(randomWalkRayIdx));
    pathStateData[jobIndex].eye_normal = make_float4(fN, 0.0f);
    pathStateData[jobIndex].pre_eye_dir = make_float4(dir, 0.0f);
    pathStateData[jobIndex].currentEye_hitData = hitData;

    path_s_t_type_pass = (s << 27) + (t << 22) + (type << 19) + pass;
    pathStateData[jobIndex].pathInfo.w = path_s_t_type_pass;

    float3 eye_pos = make_float3(pathStateData[jobIndex].data6);

    float3 light_pos = make_float3(pathStateData[jobIndex].data2);
    float3 eye2light = light_pos - eye_pos;
    float3 eye_normal = make_float3(pathStateData[jobIndex].eye_normal);
    const float dist = length(eye2light);
    eye2light = eye2light / dist;

    visibilityRays[jobIndex].O4 = make_float4(SafeOrigin(eye_pos, eye2light, eye_normal, geometryEpsilon), 0);
    visibilityRays[jobIndex].D4 = make_float4(eye2light, dist - 2 * geometryEpsilon);

    if (shadingData.IsEmissive())
    {
        float3 pre_pos = pos;
        
        float3 L = throughput * shadingData.color;

        const CoreTri& tri = (const CoreTri&)instanceTriangles[primIdx];
        const float pickProb = LightPickProb(tri.ltriIdx, pre_pos, dir, eye_pos);
        const float pdfPos = 1.0f / tri.area;
        const float p_rev = pickProb * pdfPos; // surface area

        float misWeight = 1.0f / (dE * p_rev + NKK);

        accumulatorOnePass[jobIndex] += make_float4((L*misWeight), misWeight);

        pathStateData[jobIndex].data6.w = 0;
    }
    else if (t == 1)
    {
        const uint explicitIdx = atomicAdd(&counters->contribution_explicit, 1);
        contributionBuffer_Explicit[explicitIdx] = jobIndex;

        float3 pre_pos = pos;

        float3 light2eye = eye2light;
        float length_l2e = dist;

        float bsdfPdf;
        const float3 sampledBSDF = EvaluateBSDF(shadingData, fN, T, dir * -1.0f, light2eye, bsdfPdf);

        float3 light_throughput = make_float3(pathStateData[jobIndex].data0);
        float light_pdf = pathStateData[jobIndex].data1.w;

        float3 light_normal = make_float3(pathStateData[jobIndex].light_normal);
        float light_cosTheta = fabs(dot(light2eye * -1.0f, light_normal));

        // area to solid angle: r^2 / (Area * cos)
        light_pdf *= length_l2e * length_l2e / light_cosTheta;

        float cosTheta = fabs(dot(fN, light2eye));

        float3 eye_normal = fN;
        float eye_cosTheta = fabs(dot(light2eye, eye_normal));

        float p_forward = bsdfPdf * light_cosTheta / (length_l2e * length_l2e);
        float p_rev = light_cosTheta * INVPI * eye_cosTheta / (length_l2e * length_l2e);

        float dL = pathStateData[jobIndex].data0.w;

        float misWeight = 1.0 / (dE * p_rev + 1 + dL * p_forward);
        if (bsdfPdf < EPSILON || isnan(bsdfPdf))
        {
            misWeight = 0.0f;
        }

        float3 L = throughput * sampledBSDF * light_throughput * (1.0f / light_pdf)  * cosTheta;

        pathStateData[jobIndex].L = make_float4(L * misWeight, misWeight);
    }
    else
    {
        const uint connectionIdx = atomicAdd(&counters->contribution_connection, 1);
        contributionBuffer_Connection[connectionIdx] = jobIndex;
    }
}

//  +-----------------------------------------------------------------------------+
//  |  constructionLightPos                                                            |
//  |  Entry point for the persistent constructionLightPos kernel.               LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void extendEyePath(int smcount, BiPathState* pathStateBuffer,
    Ray4* visibilityRays, Ray4* randomWalkRays, const uint R0, const uint* blueNoise,
    const float spreadAngle, const int4 screenParams, const int probePixelIdx,
    uint* eyePathBuffer, uint* contributionBuffer_Emissive, uint* contributionBuffer_Explicit,
    uint* contributionBuffer_Connection,
    float4* accumulatorOnePass, float NKK)
{
	const dim3 gridDim( NEXTMULTIPLEOF(smcount, 256 ) / 256, 1 ), blockDim( 256, 1 );
    extendEyePathKernel << < gridDim.x, 256 >> > (smcount, pathStateBuffer,
        visibilityRays, randomWalkRays,
        R0, blueNoise, spreadAngle, screenParams, probePixelIdx,eyePathBuffer,
        contributionBuffer_Emissive, contributionBuffer_Explicit,
        contributionBuffer_Connection, accumulatorOnePass, NKK);
}

// EOF
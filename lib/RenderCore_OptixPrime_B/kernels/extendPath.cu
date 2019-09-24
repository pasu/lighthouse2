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
//  |  generateEyeRaysKernel                                                      |
//  |  Generate primary rays, to be traced by Optix Prime.                  LH2'19|
//  +-----------------------------------------------------------------------------+
__global__  __launch_bounds__( 256 , 1 )
void extendPathKernel( int smcount, BiPathState* pathStateData,
    Ray4* visibilityRays, Ray4* randomWalkRays,
    const uint R0, const uint* blueNoise, const float aperture, const float imgPlaneSize,
    const float3 pos, const float3 right, const float3 up, const float3 forward, const float3 p1,
    const float spreadAngle, const int4 screenParams)
{
    int jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (jobIndex >= smcount) return;

    uint path_s_t_type_pass = pathStateData[jobIndex].pathInfo.w;

    uint pass, type, t, s;
    getPathInfo(path_s_t_type_pass, pass, s, t, type);

    const int scrhsize = screenParams.x & 0xffff;
    const int scrvsize = screenParams.x >> 16;
    const uint x = jobIndex % scrhsize;
    uint y = jobIndex / scrhsize;
    const uint sampleIndex = pass;
    y %= scrvsize;

    if((type & 0x1) == 0)
    {
        // get random numbers
        float3 posOnPixel, posOnLens;
        // depth of field camera for no filter
        float r0, r1, r2, r3;
        if (sampleIndex < 256)
        {
            r0 = blueNoiseSampler(blueNoise, x, y, sampleIndex, 4);
            r1 = blueNoiseSampler(blueNoise, x, y, sampleIndex, 5);
            r2 = blueNoiseSampler(blueNoise, x, y, sampleIndex, 6);
            r3 = blueNoiseSampler(blueNoise, x, y, sampleIndex, 7);
        }
        else
        {
            uint seed = WangHash(jobIndex + R0);
            r0 = RandomFloat(seed), r1 = RandomFloat(seed);
            r2 = RandomFloat(seed), r3 = RandomFloat(seed);
        }
        posOnPixel = p1 + ((float)x + r0) * (right / (float)scrhsize) + ((float)y + r1) * (up / (float)scrvsize);
        posOnLens = RandomPointOnLens(r2, r3, pos, aperture, right, up);
        const float3 rayDir = normalize(posOnPixel - posOnLens);

        const uint randomWalkRayIdx = atomicAdd(&counters->randomWalkRays, 1);
        randomWalkRays[randomWalkRayIdx].O4 = make_float4(posOnLens, EPSILON);
        randomWalkRays[randomWalkRayIdx].D4 = make_float4(rayDir, 1e34f);

        float4 value = make_float4(make_float3(1.0f), 0.0f);
        float3 normal = normalize(forward);
        float cosTheta = fabs(dot(normal, rayDir));

        float eye_pdf_solid = 1.0f / (imgPlaneSize * cosTheta * cosTheta * cosTheta);

        pathStateData[jobIndex].data4 = value;
        pathStateData[jobIndex].data5 = value;
        pathStateData[jobIndex].data6 = make_float4(posOnLens, eye_pdf_solid);
        pathStateData[jobIndex].data7 = make_float4(rayDir, __int_as_float(randomWalkRayIdx));
        pathStateData[jobIndex].eye_normal = make_float4(normal, 0.0f);
    }

    if (type != 0)
    {
        float3 pos, dir;
        float4 hitData;

        float d, pdf_area, pdf_solidangle;
        float3 throughput, beta;

        if (type == 1) // extend eye path
        {
            throughput = make_float3(pathStateData[jobIndex].data4);
            beta = make_float3(pathStateData[jobIndex].data5);
            pos = make_float3(pathStateData[jobIndex].data6);
            dir = make_float3(pathStateData[jobIndex].data7);
            
            d = pathStateData[jobIndex].data4.w;
            pdf_area = pathStateData[jobIndex].data5.w;
            pdf_solidangle = pathStateData[jobIndex].data6.w;

            hitData = pathStateData[jobIndex].eye_intersection;
        }
        else if (type == 2) // extend light path
        {
            throughput = make_float3(pathStateData[jobIndex].data0);
            beta = make_float3(pathStateData[jobIndex].data1);
            pos = make_float3(pathStateData[jobIndex].data2);
            dir = make_float3(pathStateData[jobIndex].data3);
            
            d = pathStateData[jobIndex].data0.w;
            pdf_area = pathStateData[jobIndex].data1.w;
            pdf_solidangle = pathStateData[jobIndex].data2.w;

            hitData = pathStateData[jobIndex].light_intersection;
        }

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
        if (sampleIndex < 256)
        {
            r4 = blueNoiseSampler(blueNoise, x, y, sampleIndex, 4);
            r5 = blueNoiseSampler(blueNoise, x, y, sampleIndex, 5);
        }
        else
        {
            uint seed = WangHash(jobIndex + R0);
            r4 = RandomFloat(seed);
            r5 = RandomFloat(seed);
        }
        const float3 bsdf = SampleBSDF(shadingData, fN, N, T, dir * -1.0f, r4, r5, R, pdf_solidangle);
       
        beta *= bsdf * fabs(dot(fN, R)) / pdf_solidangle;

        const uint randomWalkRayIdx = atomicAdd(&counters->randomWalkRays, 1);
        randomWalkRays[randomWalkRayIdx].O4 = make_float4(SafeOrigin(I, R, N, geometryEpsilon), 0);
        randomWalkRays[randomWalkRayIdx].D4 = make_float4(R, 1e34f);

        if (type == 1) // eye path
        {
            s++;

            float dE = 1.0f / pdf_area;
            if (s > 1)
            {
                float3 light_pos = make_float3(pathStateData[jobIndex].data2);
                float3 light2eye = normalize(light_pos - I);

                float bsdfPdf;
                const float3 sampledBSDF = EvaluateBSDF(shadingData, fN, T, light2eye, dir * -1.0f, bsdfPdf);

                float3 normal = make_float3(pathStateData[jobIndex].eye_normal);
                float light_p = bsdfPdf * fabs(dot(normal, dir)) / (HIT_T * HIT_T);

                dE = (1.0f + light_p * d) / pdf_area;
            }

            pathStateData[jobIndex].data4 = make_float4(throughput,dE);
            pathStateData[jobIndex].data5 = make_float4(beta, pdf_area);;
            pathStateData[jobIndex].data6 = make_float4(I, pdf_solidangle);
            pathStateData[jobIndex].data7 = make_float4(R, __int_as_float(randomWalkRayIdx));
            pathStateData[jobIndex].eye_normal = make_float4(fN, 0.0f);
            pathStateData[jobIndex].pre_eye_dir = make_float4(dir, 0.0f);
            pathStateData[jobIndex].currentEye_hitData = hitData;
        }
        else if (type == 2) // light path
        {
            t++;

            float3 eye_pos = make_float3(pathStateData[jobIndex].data6);
            float3 eye2light = normalize(eye_pos - I);

            float bsdfPdf;
            const float3 sampledBSDF = EvaluateBSDF(shadingData, fN, T, eye2light, dir * -1.0f, bsdfPdf);

            float3 normal = make_float3(pathStateData[jobIndex].light_normal);
            float eye_p = bsdfPdf * fabs(dot(normal, dir)) / (HIT_T * HIT_T);
            float dL = (1.0f + eye_p * d) / pdf_area;

            pathStateData[jobIndex].data0 = make_float4(throughput, dL);
            pathStateData[jobIndex].data1 = make_float4(beta, pdf_area);
            pathStateData[jobIndex].data2 = make_float4(I, pdf_solidangle);
            pathStateData[jobIndex].data3 = make_float4(R, __int_as_float(randomWalkRayIdx));
            pathStateData[jobIndex].light_normal = make_float4(fN, 0.0f);
            pathStateData[jobIndex].pre_light_dir = make_float4(dir, 0.0f);
            pathStateData[jobIndex].currentLight_hitData = hitData;
        }
    }

    path_s_t_type_pass = (s << 27) + (t << 22) + (type << 19) + pass;
    pathStateData[jobIndex].pathInfo.w = path_s_t_type_pass;

    float3 eye_pos = make_float3(pathStateData[jobIndex].data6);
    float3 eye2light = normalize(make_float3(pathStateData[jobIndex].data2) - eye_pos);
    float3 eye_normal = make_float3(pathStateData[jobIndex].eye_normal);
    const float dist = length(eye_pos - eye2light);

    visibilityRays[jobIndex].O4 = make_float4(SafeOrigin(eye_pos, eye2light, eye_normal, geometryEpsilon), 0);
    visibilityRays[jobIndex].D4 = make_float4(eye2light, dist - 2 * geometryEpsilon);
}

//  +-----------------------------------------------------------------------------+
//  |  constructionLightPos                                                            |
//  |  Entry point for the persistent constructionLightPos kernel.               LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void extendPath(int smcount, BiPathState* pathStateBuffer,
    Ray4* visibilityRays, Ray4* randomWalkRays,
    const uint R0, const uint* blueNoise, const float lensSize, const float imgPlaneSize,
    const float3 camPos, const float3 right, const float3 up, const float3 forward, const float3 p1,
    const float spreadAngle, const int4 screenParams)
{
	const dim3 gridDim( NEXTMULTIPLEOF(smcount, 256 ) / 256, 1 ), blockDim( 256, 1 );
    extendPathKernel << < gridDim.x, 256 >> > (smcount, pathStateBuffer,
        visibilityRays, randomWalkRays,
        R0, blueNoise, lensSize, imgPlaneSize,
        camPos, right, up, forward, p1, spreadAngle, screenParams);
}

// EOF
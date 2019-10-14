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
void extendLightPathKernel(int smcount, BiPathState* pathStateData,
    Ray4* visibilityRays, Ray4* randomWalkRays, const uint R0, const uint* blueNoise,
    const float3 cam_pos, const float spreadAngle, const int4 screenParams, 
    uint* lightPathBuffer, uint* contributionBuffer_Photon)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= counters->extendLightPath) return;

    int jobIndex = lightPathBuffer[gid];

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

        
    throughput = make_float3(pathStateData[jobIndex].data0);
    beta = make_float3(pathStateData[jobIndex].data1);
    pos = make_float3(pathStateData[jobIndex].data2);
    dir = make_float3(pathStateData[jobIndex].data3);

    d = pathStateData[jobIndex].data0.w;
    pdf_area = pathStateData[jobIndex].data1.w;
    pdf_solidangle = pathStateData[jobIndex].data2.w;

    hitData = pathStateData[jobIndex].light_intersection;

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

    float test = pdf_solidangle;

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
    const float3 bsdf = SampleBSDF(shadingData, fN, N, T, dir * -1.0f, r4, r5, R, pdf_solidangle,type);
       
    beta *= bsdf * fabs(dot(fN, R)) / pdf_solidangle;

    // correct shading normal when it is importance
    float shading_normal_num = fabs(dot(dir, fN)) * fabs(dot(R, N));
    float shading_normal_denom = fabs(dot(dir, N)) * fabs(dot(R, fN));

    if (shading_normal_denom != 0)
    {
        beta *= (shading_normal_num / shading_normal_denom);
    }

    const uint randomWalkRayIdx = atomicAdd(&counters->randomWalkRays, 1);
    randomWalkRays[randomWalkRayIdx].O4 = make_float4(SafeOrigin(I, R, N, geometryEpsilon), 0);
    randomWalkRays[randomWalkRayIdx].D4 = make_float4(R, 1e34f);

        
    t++;

    float3 eye_pos = cam_pos;
    float3 eye2lightU = normalize(eye_pos - I);

    float bsdfPdf;
    const float3 sampledBSDF = EvaluateBSDF(shadingData, fN, T, eye2lightU, dir * -1.0f, bsdfPdf);

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

    path_s_t_type_pass = (s << 27) + (t << 22) + (type << 19) + pass;
    pathStateData[jobIndex].pathInfo.w = path_s_t_type_pass;

    float3 light_pos = make_float3(pathStateData[jobIndex].data2);
    float3 eye2light = light_pos - eye_pos;
    float3 eye_normal = make_float3(pathStateData[jobIndex].eye_normal);
    const float dist = length(eye_pos - eye2light);
    eye2light = eye2light / dist;

    visibilityRays[jobIndex].O4 = make_float4(SafeOrigin(eye_pos, eye2light, eye_normal, geometryEpsilon), 0);
    visibilityRays[jobIndex].D4 = make_float4(eye2light, dist - 2 * geometryEpsilon);

    const uint photonIdx = atomicAdd(&counters->contribution_photon, 1);
    contributionBuffer_Photon[photonIdx] = jobIndex;
}

//  +-----------------------------------------------------------------------------+
//  |  constructionLightPos                                                            |
//  |  Entry point for the persistent constructionLightPos kernel.               LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void extendLightPath(int smcount, BiPathState* pathStateBuffer,
    Ray4* visibilityRays, Ray4* randomWalkRays, const uint R0, const uint* blueNoise,
    const float3 camPos,const float spreadAngle, const int4 screenParams,
    uint* lightPathBuffer, uint* contributionBuffer_Photon)
{
	const dim3 gridDim( NEXTMULTIPLEOF(smcount, 256 ) / 256, 1 ), blockDim( 256, 1 );
    extendLightPathKernel << < gridDim.x, 256 >> > (smcount, pathStateBuffer,
        visibilityRays, randomWalkRays,
        R0, blueNoise, camPos, spreadAngle, screenParams, 
        lightPathBuffer, contributionBuffer_Photon);
}

// EOF
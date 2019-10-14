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

//  +-----------------------------------------------------------------------------+
//  |  generateEyeRaysKernel                                                      |
//  |  Generate primary rays, to be traced by Optix Prime.                  LH2'19|
//  +-----------------------------------------------------------------------------+
__global__  __launch_bounds__( 256 , 1 )
void connectionPath_EmissiveKernel(int smcount, float NKK, BiPathState* pathStateData,
    const float spreadAngle, float4* accumulatorOnePass,
    float4* weightMeasureBuffer, const int4 screenParams,
    uint* contributionBuffer_Emissive)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= counters->contribution_emissive) return;

    int jobIndex = contributionBuffer_Emissive[gid];

    const int scrhsize = screenParams.x & 0xffff;
    const int scrvsize = screenParams.x >> 16;

    const uint x_line = jobIndex % scrhsize;
    uint y_line = jobIndex / scrhsize;

    uint path_s_t_type_pass = pathStateData[jobIndex].pathInfo.w;

    uint pass, type, t, s;
    getPathInfo(path_s_t_type_pass,pass,s,t,type);

    const float3 empty_color = make_float3(0.0f);
    float3 L = empty_color;
    float misWeight = 0.0f;

    float4 hitData = pathStateData[jobIndex].currentEye_hitData;
    float3 dir = make_float3(pathStateData[jobIndex].pre_eye_dir);

    float3 throughput = make_float3(pathStateData[jobIndex].data4);
    float3 beta = make_float3(pathStateData[jobIndex].data5);
    float3 eye_pos = make_float3(pathStateData[jobIndex].data6);
    float3 pre_pos = eye_pos - dir * HIT_T;
    float dE = pathStateData[jobIndex].data4.w;

    const int prim = __float_as_int(hitData.z);
    const int primIdx = prim == -1 ? prim : (prim & 0xfffff);

    const CoreTri4* instanceTriangles = (const CoreTri4*)instanceDescriptors[INSTANCEIDX].triangles;

    ShadingData shadingData;
    float3 N, iN, fN, T;

    const float coneWidth = spreadAngle * HIT_T;
    GetShadingData(dir, HIT_U, HIT_V, coneWidth, instanceTriangles[primIdx], INSTANCEIDX, shadingData, N, iN, fN, T);

    L = throughput * shadingData.color;

    const CoreTri& tri = (const CoreTri&)instanceTriangles[primIdx];
    const float pickProb = LightPickProb(tri.ltriIdx, pre_pos, dir, eye_pos);
    const float pdfPos = 1.0f / tri.area;

    const float p_rev = pickProb * pdfPos; // surface area

    misWeight = 1.0f / (dE * p_rev + NKK);
    weightMeasureBuffer[jobIndex].x += misWeight;
    
    accumulatorOnePass[jobIndex] += make_float4((L*misWeight), misWeight);
}

//  +-----------------------------------------------------------------------------+
//  |  constructionLightPos                                                            |
//  |  Entry point for the persistent constructionLightPos kernel.               LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void connectionPath_Emissive(int smcount, float NKK, BiPathState* pathStateData,
    const float spreadAngle, float4* accumulatorOnePass,
    float4* weightMeasureBuffer, const int4 screenParams,
    uint* contributionBuffer_Emissive)
{
	const dim3 gridDim( NEXTMULTIPLEOF(smcount, 256 ) / 256, 1 ), blockDim( 256, 1 );
    connectionPath_EmissiveKernel << < gridDim.x, 256 >> > (smcount, NKK, pathStateData,
        spreadAngle, accumulatorOnePass,weightMeasureBuffer,screenParams,
        contributionBuffer_Emissive);
}

// EOF
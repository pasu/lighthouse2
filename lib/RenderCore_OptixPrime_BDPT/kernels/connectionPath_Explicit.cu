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
void connectionPath_ExplicitKernel(int smcount, BiPathState* pathStateData,
    uint* visibilityHitBuffer, const float spreadAngle, 
    float4* accumulatorOnePass, float4* weightMeasureBuffer,
    const int4 screenParams, uint* contributionBuffer_Explicit)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= counters->contribution_explicit) return;

    int jobIndex = contributionBuffer_Explicit[gid];

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

    const uint occluded = visibilityHitBuffer[jobIndex >> 5] & (1 << (jobIndex & 31));

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

    float3 light_pos = make_float3(pathStateData[jobIndex].data2);
    float3 light2eye = light_pos - eye_pos;
    float length_l2e = length(light2eye);
    light2eye /= length_l2e;

    float bsdfPdf;
    const float3 sampledBSDF = EvaluateBSDF(shadingData, fN, T, dir * -1.0f, light2eye, bsdfPdf);

    float3 light_throughput = make_float3(pathStateData[jobIndex].data0);
    float light_pdf = pathStateData[jobIndex].data1.w;

    float3 light_normal = make_float3(pathStateData[jobIndex].light_normal);
    float light_cosTheta = fabs(dot(light2eye * -1.0f, light_normal));

    // area to solid angle: r^2 / (Area * cos)
    light_pdf *= length_l2e * length_l2e / light_cosTheta;

    float cosTheta = fabs(dot(fN, light2eye));

    float3 eye_normal = make_float3(pathStateData[jobIndex].eye_normal);
    float eye_cosTheta = fabs(dot(light2eye, eye_normal));

    float p_forward = bsdfPdf * light_cosTheta / (length_l2e * length_l2e);
    float p_rev = light_cosTheta * INVPI * eye_cosTheta / (length_l2e * length_l2e);

    float dL = pathStateData[jobIndex].data0.w;

    misWeight = 1.0 / (dE * p_rev + 1 + dL * p_forward);
    weightMeasureBuffer[jobIndex].y += misWeight;

    if (!occluded)
    {
        L = throughput * sampledBSDF * light_throughput * (1.0f / light_pdf)  * cosTheta;
    }

    if (bsdfPdf < EPSILON || isnan(bsdfPdf))
    {
        L = empty_color;
    }
                
    accumulatorOnePass[jobIndex] += make_float4((L*misWeight), misWeight);
}

//  +-----------------------------------------------------------------------------+
//  |  constructionLightPos                                                            |
//  |  Entry point for the persistent constructionLightPos kernel.               LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void connectionPath_Explicit(int smcount, BiPathState* pathStateData,
    uint* visibilityHitBuffer, const float spreadAngle,
    float4* accumulatorOnePass, float4* weightMeasureBuffer,
    const int4 screenParams, uint* contributionBuffer_Explicit)
{
	const dim3 gridDim( NEXTMULTIPLEOF(smcount, 256 ) / 256, 1 ), blockDim( 256, 1 );
    connectionPath_ExplicitKernel << < gridDim.x, 256 >> > (smcount, pathStateData,
        visibilityHitBuffer,spreadAngle,accumulatorOnePass,weightMeasureBuffer,
        screenParams,contributionBuffer_Explicit);
}

// EOF
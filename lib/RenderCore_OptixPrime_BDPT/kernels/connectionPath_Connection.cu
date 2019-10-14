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
void connectionPath_ConnectionKernel(int smcount, BiPathState* pathStateData,
    uint* visibilityHitBuffer,const float spreadAngle, 
    float4* accumulatorOnePass, 
    const int4 screenParams, uint* contributionBuffer_Connection)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= counters->contribution_connection) return;

    int jobIndex = contributionBuffer_Connection[gid];

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

    const CoreTri4* instanceTriangles_light = (const CoreTri4*)instanceDescriptors[INSTANCEIDX].triangles;

    ShadingData shadingData;
    float3 N, iN, fN, T;

    const float coneWidth = spreadAngle * HIT_T;
    GetShadingData(dir, HIT_U, HIT_V, coneWidth, instanceTriangles_light[primIdx], INSTANCEIDX, shadingData, N, iN, fN, T);

    float3 light_pos = make_float3(pathStateData[jobIndex].data2);
    float3 light2eye = light_pos - eye_pos;
    float length_l2e = length(light2eye);
    light2eye /= length_l2e;

    float eye_bsdfPdf;
    const float3 sampledBSDF_s = EvaluateBSDF(shadingData, fN, T, dir * -1.0f, light2eye, eye_bsdfPdf);

    hitData = pathStateData[jobIndex].currentLight_hitData;

    float3 dir_light = make_float3(pathStateData[jobIndex].pre_light_dir);

    const int prim_light = __float_as_int(hitData.z);
    const int primIdx_light = prim_light == -1 ? prim_light : (prim_light & 0xfffff);
    int idx = (prim_light >> 20);

    const CoreTri4* instanceTriangles_eye = (const CoreTri4*)instanceDescriptors[idx].triangles;

    ShadingData shadingData_light;
    float3 N_light, iN_light, fN_light, T_light;
    
    GetShadingData(dir_light, HIT_U, HIT_V, coneWidth, instanceTriangles_eye[primIdx_light],
        idx, shadingData_light, N_light, iN_light, fN_light, T_light);

    float light_bsdfPdf;
    float3 sampledBSDF_t = EvaluateBSDF(shadingData_light, fN_light, T_light,
        dir_light * -1.0f, light2eye * -1.0f, light_bsdfPdf);

    float shading_normal_num = fabs(dot(dir_light, fN_light)) * fabs(dot(light2eye, N_light));
    float shading_normal_denom = fabs(dot(dir_light, N_light)) * fabs(dot(light2eye, fN_light));

    if (shading_normal_denom != 0)
    {
        sampledBSDF_t *= (shading_normal_num / shading_normal_denom);
    }

    float3 throughput_light = make_float3(pathStateData[jobIndex].data0);

    // fabs keep safety
    float cosTheta_eye = fabs(dot(fN, light2eye));
    float cosTheta_light = fabs(dot(fN_light, light2eye* -1.0f));
    float G = cosTheta_eye * cosTheta_light / (length_l2e * length_l2e);

    if (!occluded)
    {
        L = throughput * sampledBSDF_s * sampledBSDF_t * throughput_light * G;
    }

    float p_forward = eye_bsdfPdf * cosTheta_light / (length_l2e * length_l2e);
    float p_rev = light_bsdfPdf * cosTheta_eye / (length_l2e * length_l2e);

    float dL = pathStateData[jobIndex].data0.w;

    misWeight = 1.0 / (dE * p_rev + 1 + dL * p_forward);

    if (eye_bsdfPdf < EPSILON || isnan(eye_bsdfPdf)
        || light_bsdfPdf < EPSILON || isnan(light_bsdfPdf))
    {
        L = empty_color;
    }
            
    accumulatorOnePass[jobIndex] += make_float4((L*misWeight), misWeight);
}

//  +-----------------------------------------------------------------------------+
//  |  constructionLightPos                                                            |
//  |  Entry point for the persistent constructionLightPos kernel.               LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void connectionPath_Connection(int smcount, BiPathState* pathStateData,
    uint* visibilityHitBuffer, const float spreadAngle,
    float4* accumulatorOnePass,
    const int4 screenParams, uint* contributionBuffer_Connection)
{
	const dim3 gridDim( NEXTMULTIPLEOF(smcount, 256 ) / 256, 1 ), blockDim( 256, 1 );
    connectionPath_ConnectionKernel << < gridDim.x, 256 >> > (smcount, pathStateData,
        visibilityHitBuffer,spreadAngle,accumulatorOnePass,
        screenParams,contributionBuffer_Connection);
}

// EOF
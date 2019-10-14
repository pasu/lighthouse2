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
void connectionPathKernel(int smcount, float NKK, float scene_area, BiPathState* pathStateData,
    const Intersection* randomWalkHitBuffer,
    float4* accumulatorOnePass, uint* constructLightBuffer,
    float4* weightMeasureBuffer, const int4 screenParams,
    uint* constructEyeBuffer, uint* eyePathBuffer, uint* lightPathBuffer)
{
    int jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (jobIndex >= smcount) return;

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
    
    int eye_hit = -1;
    int eye_hit_idx = __float_as_int(pathStateData[jobIndex].data7.w);
    float eye_pdf = pathStateData[jobIndex].data6.w;
    if (eye_pdf < EPSILON || isnan(eye_pdf))
    {
        eye_hit = -1;
        pathStateData[jobIndex].data7.w = __int_as_float(-1);
    }
    else if (eye_hit_idx > -1)
    {
        const Intersection hd = randomWalkHitBuffer[eye_hit_idx];

        eye_hit = hd.triid;

        const float4 hitData = make_float4(hd.u, hd.v, __int_as_float(hd.triid + (hd.triid == -1 ? 0 : (hd.instid << 20))), hd.t);
        pathStateData[jobIndex].eye_intersection = hitData;

        pathStateData[jobIndex].data7.w = __int_as_float(-1);
    }

    int light_hit = -1;
    int light_hit_idx = __float_as_int(pathStateData[jobIndex].data3.w);
    float light_pdf_test = pathStateData[jobIndex].data2.w;
    if (light_pdf_test < EPSILON || isnan(light_pdf_test))
    {
        light_hit = -1;
        pathStateData[jobIndex].data3.w = __int_as_float(-1);
    }
    else if (light_hit_idx > -1)
    {
        const Intersection hd = randomWalkHitBuffer[light_hit_idx];
        light_hit = hd.triid;
        const float4 hitData = make_float4(hd.u, hd.v, __int_as_float(hd.triid + (hd.triid == -1 ? 0 : (hd.instid << 20))), hd.t);

        pathStateData[jobIndex].light_intersection = hitData;
        pathStateData[jobIndex].data3.w = __int_as_float(-1);
    }
    else
    {
        const float4 hitData = pathStateData[jobIndex].light_intersection;

        const int prim = __float_as_int(hitData.z);
        const int primIdx = prim == -1 ? prim : (prim & 0xfffff);

        light_hit = primIdx;
    }

    const uint MAX__LENGTH_E = 3;
    const uint MAX__LENGTH_L = 5;

    if (eye_hit != -1 && s < MAX__LENGTH_E)
    {
        type = EXTEND_EYEPATH;
        const uint eyePIdx = atomicAdd(&counters->extendEyePath, 1);
        eyePathBuffer[eyePIdx] = jobIndex;
    }
    else if (light_hit != -1 && t < MAX__LENGTH_L)
    {
        type = EXTEND_LIGHTPATH;

        const uint eyeIdx = atomicAdd(&counters->constructionEyePos, 1);
        constructEyeBuffer[eyeIdx] = jobIndex;

        const uint lightPIdx = atomicAdd(&counters->extendLightPath, 1);
        lightPathBuffer[lightPIdx] = jobIndex;
    }
    else
    {
        const uint constructLight = atomicAdd(&counters->constructionLightPos, 1);
        constructLightBuffer[constructLight] = jobIndex;
    }

    if (eye_hit == -1 && type != EXTEND_LIGHTPATH)
    {
        float3 hit_dir = make_float3(pathStateData[jobIndex].data7);
        float3 background = make_float3(SampleSkydome(hit_dir, s+1));

        // hit miss : beta 
        float3 beta = make_float3(pathStateData[jobIndex].data5);
        float3 contribution = beta * background;

        CLAMPINTENSITY; // limit magnitude of thoughput vector to combat fireflies
        FIXNAN_FLOAT3(contribution);

        float dE = pathStateData[jobIndex].data4.w;
        misWeight = 1.0f;// / (dE * (1.0f / (scene_area)) + NKK);

        accumulatorOnePass[jobIndex] += make_float4((contribution * misWeight), misWeight);
    }

    path_s_t_type_pass = (s << 27) + (t << 22) + (type << 19) + pass;
    pathStateData[jobIndex].pathInfo.w = path_s_t_type_pass;
}

//  +-----------------------------------------------------------------------------+
//  |  constructionLightPos                                                            |
//  |  Entry point for the persistent constructionLightPos kernel.               LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void connectionPath(int smcount, float NKK, float scene_area, 
    BiPathState* pathStateData, const Intersection* randomWalkHitBuffer,
    float4* accumulatorOnePass, uint* constructLightBuffer,
    float4* weightMeasureBuffer, const int4 screenParams,
    uint* constructEyeBuffer, uint* eyePathBuffer, uint* lightPathBuffer)
{
	const dim3 gridDim( NEXTMULTIPLEOF(smcount, 256 ) / 256, 1 ), blockDim( 256, 1 );
    connectionPathKernel << < gridDim.x, 256 >> > (smcount, NKK, scene_area, 
        pathStateData, randomWalkHitBuffer, accumulatorOnePass, 
        constructLightBuffer, weightMeasureBuffer, screenParams,
        constructEyeBuffer, eyePathBuffer, lightPathBuffer);
}

// EOF
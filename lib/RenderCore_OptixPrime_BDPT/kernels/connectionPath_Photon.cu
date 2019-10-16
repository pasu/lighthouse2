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

LH2_DEVFUNC void Sample_Wi(const float aperture, const float imgPlaneSize, const float3 eye_pos,
    const float3 forward, const float3 light_pos, const float focalDistance,
    const float3 p1, const float3 right, const float3 up,
    float3& throughput, float& pdf, float& u, float& v)
{
    throughput = make_float3(0.0f);
    pdf = 0.0f;

    float3 dir = light_pos - eye_pos;
    float dist = length(dir);

    dir /= dist;

    float cosTheta = dot(normalize(forward), dir);

    // check direction
    if (cosTheta <= 0)
    {
        return;
    }

    float x_length = length(right);
    float y_length = length(up);

    float distance = focalDistance / cosTheta;

    float3 raster_pos = eye_pos + distance * dir;
    float3 pos2p1 = raster_pos - p1;

    float3 unit_up = up / y_length;
    float3 unit_right = right / x_length;

    float x_offset = dot(unit_right, pos2p1);
    float y_offset = dot(unit_up, pos2p1);

    // check view fov
    if (x_offset<0 || x_offset > x_length
        || y_offset<0 || y_offset > y_length)
    {
        //printf("%f,%f,%f,%f\n", x_offset, x_length,y_offset, y_length);
        return;
    }

    //printf("in raster\n");

    u = x_offset / x_length;
    v = y_offset / y_length;

    float cos2Theta = cosTheta * cosTheta;
    float lensArea = aperture != 0 ? aperture * aperture * PI : 1;
    lensArea = 1.0f; // because We / pdf
    float We = 1.0f / (imgPlaneSize * lensArea * cos2Theta * cos2Theta);

    throughput = make_float3(We);
    pdf = dist * dist / (cosTheta * lensArea);
}

//  +-----------------------------------------------------------------------------+
//  |  generateEyeRaysKernel                                                      |
//  |  Generate primary rays, to be traced by Optix Prime.                  LH2'19|
//  +-----------------------------------------------------------------------------+
__global__  __launch_bounds__( 256 , 1 )
void connectionPath_PhotonKernel(int smcount, BiPathState* pathStateData,
    uint* visibilityHitBuffer,const float aperture, const float imgPlaneSize, 
    const float3 forward, const float focalDistance, const float3 p1, 
    const float3 right, const float3 up, const float spreadAngle, 
    float4* accumulatorOnePass, const int4 screenParams,
    const float3 camPos, 
    uint* contributionBuffer_Photon, const int probePixelIdx)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= counters->contribution_photon) return;

    int jobIndex = contributionBuffer_Photon[gid];

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

    float3 light_pos = make_float3(pathStateData[jobIndex].data2);
    float3 eye_pos = camPos;

    float3 light2eye = eye_pos - light_pos;
    float length_l2e = length(light2eye);
    light2eye /= length_l2e;

    float3 throughput_eye;
    float pdf_eye;
    float u, v;
    Sample_Wi(aperture, imgPlaneSize, eye_pos, forward, light_pos,
        focalDistance, p1, right, up, throughput_eye, pdf_eye, u, v);

    if (pdf_eye > EPSILON)
    {
        float4 hitData = pathStateData[jobIndex].currentLight_hitData;
        float3 dir = make_float3(pathStateData[jobIndex].pre_light_dir);

        const int prim = __float_as_int(hitData.z);
        const int primIdx = prim == -1 ? prim : (prim & 0xfffff);

        const CoreTri4* instanceTriangles = (const CoreTri4*)instanceDescriptors[INSTANCEIDX].triangles;

        ShadingData shadingData;
        float3 N, iN, fN, T;

        const float coneWidth = spreadAngle * HIT_T;
        GetShadingData(dir, HIT_U, HIT_V, coneWidth, instanceTriangles[primIdx], INSTANCEIDX, shadingData, N, iN, fN, T);

        float bsdfPdf;
        const float3 sampledBSDF = EvaluateBSDF(shadingData, fN, T, dir * -1.0f, light2eye, bsdfPdf);

        float3 light_throught = make_float3(pathStateData[jobIndex].data0);
        float cosTheta = fabs(dot(fN, light2eye));

        float eye_cosTheta = fabs(dot(normalize(forward), light2eye * -1.0f));
        float eye_pdf_solid = 1.0f / (imgPlaneSize * eye_cosTheta * eye_cosTheta * eye_cosTheta);
        float p_forward = eye_pdf_solid * cosTheta / (length_l2e * length_l2e);

        float dL = pathStateData[jobIndex].data0.w;

        misWeight = 1.0f / (1 + dL * p_forward);

        if (!occluded)
        {
            uint x = (scrhsize * u + 0.5);
            uint y = (scrvsize * v + 0.5);
            uint idx = y * scrhsize + x;

            L = light_throught * sampledBSDF * (throughput_eye / pdf_eye) * cosTheta;
            //misWeight = 1.0f;
            float4 res_color = make_float4((L*misWeight), misWeight);

            /*
            if (idx == probePixelIdx)
            {
                printf("%f,%f,%f,%f\n", L.x*misWeight, light_throught.x, sampledBSDF.y, 1.0f/ pdf_eye);
            }
            */
            

            atomicAdd(&(accumulatorOnePass[idx].x), res_color.x);
            atomicAdd(&(accumulatorOnePass[idx].y), res_color.y);
            atomicAdd(&(accumulatorOnePass[idx].z), res_color.z);
            atomicAdd(&(accumulatorOnePass[idx].w), res_color.w);

            const uint pm_idx = atomicAdd(&counters->photomappings, 1);
        }
    }    
}

//  +-----------------------------------------------------------------------------+
//  |  constructionLightPos                                                            |
//  |  Entry point for the persistent constructionLightPos kernel.               LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void connectionPath_Photon(int smcount, BiPathState* pathStateData,
    uint* visibilityHitBuffer, const float aperture, const float imgPlaneSize,
    const float3 forward, const float focalDistance, const float3 p1,
    const float3 right, const float3 up, const float spreadAngle,
    float4* accumulatorOnePass, const int4 screenParams,
    const float3 camPos,
    uint* contributionBuffer_Photon, const int probePixelIdx)
{
	const dim3 gridDim( NEXTMULTIPLEOF(smcount, 256 ) / 256, 1 ), blockDim( 256, 1 );
    connectionPath_PhotonKernel << < gridDim.x, 256 >> > (smcount, pathStateData,
        visibilityHitBuffer, aperture, imgPlaneSize,
        forward, focalDistance, p1, right, up, spreadAngle, accumulatorOnePass, 
        screenParams, camPos, contributionBuffer_Photon, probePixelIdx);
}

// EOF
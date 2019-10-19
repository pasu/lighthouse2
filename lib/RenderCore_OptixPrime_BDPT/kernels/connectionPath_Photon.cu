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

    const uint occluded = visibilityHitBuffer[jobIndex >> 5] & (1 << (jobIndex & 31));

    if (!occluded)
    {
        uint idx = __float_as_uint(pathStateData[jobIndex].L.w);

        float4 res_color = pathStateData[jobIndex].L;

        atomicAdd(&(accumulatorOnePass[idx].x), res_color.x);
        atomicAdd(&(accumulatorOnePass[idx].y), res_color.y);
        atomicAdd(&(accumulatorOnePass[idx].z), res_color.z);
        //atomicAdd(&(accumulatorOnePass[idx].w), 1.0f);
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
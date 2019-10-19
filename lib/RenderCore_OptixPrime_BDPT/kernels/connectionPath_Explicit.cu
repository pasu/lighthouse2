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
    float4* accumulatorOnePass,
    const int4 screenParams, uint* contributionBuffer_Explicit,
    const int probePixelIdx)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= counters->contribution_explicit) return;

    int jobIndex = contributionBuffer_Explicit[gid];

    const uint occluded = visibilityHitBuffer[jobIndex >> 5] & (1 << (jobIndex & 31));

    if (!occluded)
    {
        accumulatorOnePass[jobIndex] += pathStateData[jobIndex].L;
    }
}

//  +-----------------------------------------------------------------------------+
//  |  constructionLightPos                                                            |
//  |  Entry point for the persistent constructionLightPos kernel.               LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void connectionPath_Explicit(int smcount, BiPathState* pathStateData,
    uint* visibilityHitBuffer, const float spreadAngle,
    float4* accumulatorOnePass,
    const int4 screenParams, uint* contributionBuffer_Explicit,
    const int probePixelIdx)
{
	const dim3 gridDim( NEXTMULTIPLEOF(smcount, 256 ) / 256, 1 ), blockDim( 256, 1 );
    connectionPath_ExplicitKernel << < gridDim.x, 256 >> > (smcount, pathStateData,
        visibilityHitBuffer,spreadAngle,accumulatorOnePass,
        screenParams,contributionBuffer_Explicit, probePixelIdx);
}

// EOF
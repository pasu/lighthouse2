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

//  +-----------------------------------------------------------------------------+
//  |  generateEyeRaysKernel                                                      |
//  |  Generate primary rays, to be traced by Optix Prime.                  LH2'19|
//  +-----------------------------------------------------------------------------+
__global__  __launch_bounds__( 256 , 1 )
void connectionPathKernel(int smcount, BiPathState* pathStateData,
    const Intersection* randomWalkHitBuffer, uint* visibilityHitBuffer,
    float4* accumulatorOnePass)
{
    int jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (jobIndex >= smcount) return;

    uint path_s_t_type_pass = pathStateData[jobIndex].pathInfo.w;

    uint pass, type, t, s;
    getPathInfo(path_s_t_type_pass,pass,s,t,type);

    const uint occluded = visibilityHitBuffer[jobIndex >> 5] & (1 << (jobIndex & 31));
    if (!occluded)
    {
        // Sample a point on the camera and connect it to the light subpath
        if (s == 0)
        {            

        }
    }

}

//  +-----------------------------------------------------------------------------+
//  |  constructionLightPos                                                            |
//  |  Entry point for the persistent constructionLightPos kernel.               LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void connectionPath(int smcount, BiPathState* pathStateBuffer,
    const Intersection* randomWalkHitBuffer, uint* visibilityHitBuffer,
    float4* accumulatorOnePass)
{
	const dim3 gridDim( NEXTMULTIPLEOF(smcount, 256 ) / 256, 1 ), blockDim( 256, 1 );
    connectionPathKernel << < gridDim.x, 256 >> > (smcount, pathStateBuffer,
        randomWalkHitBuffer,visibilityHitBuffer,accumulatorOnePass);
}

// EOF
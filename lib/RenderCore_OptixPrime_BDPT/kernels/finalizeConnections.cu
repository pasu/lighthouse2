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
//  |  constructionLightPosKernel                                                      |
//  |  Generate the first vertex of the light path including pos and direction.                  LH2'19|
//  +-----------------------------------------------------------------------------+
__global__  __launch_bounds__( 256 , 1 )
void finalizeConnectionsKernel(int smcount, float4* accumulatorOnePass, 
    float4* accumulator)
{
    int jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (jobIndex >= smcount) return;

    accumulator[jobIndex] += accumulatorOnePass[jobIndex];
    accumulator[jobIndex].w += 1.0f;
}

//  +-----------------------------------------------------------------------------+
//  |  constructionLightPos                                                            |
//  |  Entry point for the persistent constructionLightPos kernel.               LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void finalizeConnections( int smcount, float4* accumulatorOnePass, 
    float4* accumulator)
{
	const dim3 gridDim( NEXTMULTIPLEOF(smcount, 256 ) / 256, 1 ), blockDim( 256, 1 );
    finalizeConnectionsKernel << < gridDim.x, 256 >> > (smcount,
        accumulatorOnePass, accumulator);
}

// EOF
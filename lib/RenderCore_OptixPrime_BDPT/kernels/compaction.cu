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
void compactionPathKernel(int smcount,
    BiPathState* pathStateData, BiPathState* pathStateDataOut,
    uint ext_Eye, uint ext_Light)
{
    int jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (jobIndex >= smcount) return;

    uint path_s_t_type_pass = __float_as_uint(pathStateData[jobIndex].eye_normal.w);
    
    uint pass, type, t, s;
    getPathInfo(path_s_t_type_pass,pass,s,t,type);

    //copyPathState(pathStateData[jobIndex], pathStateDataOut[jobIndex]);
    
    if (type == EXTEND_EYEPATH)
    {
        const uint eyeIdx = atomicAdd(&counters->ext_Eye_idx, 1);
        copyPathState(pathStateData[jobIndex], pathStateDataOut[ext_Light+eyeIdx]);
    }
    else if (type == EXTEND_LIGHTPATH)
    {
        const uint lightIdx = atomicAdd(&counters->ext_Light_idx, 1);
        copyPathState(pathStateData[jobIndex], pathStateDataOut[lightIdx]);
    }
    
}

//  +-----------------------------------------------------------------------------+
//  |  constructionLightPos                                                            |
//  |  Entry point for the persistent constructionLightPos kernel.               LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void compactionPath(int smcount,
    BiPathState* pathStateData, BiPathState* pathStateDataOut,
    uint ext_Eye, uint ext_Light)
{
	const dim3 gridDim( NEXTMULTIPLEOF(smcount, 256 ) / 256, 1 ), blockDim( 256, 1 );
    compactionPathKernel << < gridDim.x, 256 >> > (smcount,
        pathStateData, pathStateDataOut, ext_Eye, ext_Light);
}

// EOF
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
void constructionLightPosKernel(int smcount, float NKK,uint* constructLightBuffer, float4* pathStateData, const uint R0, const uint* blueNoise, const int4 screenParams)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= smcount) return;

    int jobIndex = constructLightBuffer[gid];

    const int scrhsize = screenParams.x & 0xffff;
    const int scrvsize = screenParams.x >> 16;
    const uint x = jobIndex % scrhsize;
    uint y = jobIndex / scrhsize;

    const uint sampleIdx = __float_as_int(pathStateData[jobIndex].y);

    float r0, r1,r2,r3;

    if (sampleIdx < 256)
    {
        r0 = blueNoiseSampler(blueNoise, x, y, sampleIdx, 0);
        r1 = blueNoiseSampler(blueNoise, x, y, sampleIdx, 1);
        r2 = blueNoiseSampler(blueNoise, x, y, sampleIdx, 2);
        r3 = blueNoiseSampler(blueNoise, x, y, sampleIdx, 3);
    }
    else
    {
        uint seed = WangHash(jobIndex + R0);

        r0 = RandomFloat(seed);
        r1 = RandomFloat(seed);
        r2 = RandomFloat(seed);
        r3 = RandomFloat(seed);
    }

    float3 normal, throughput, lightDir;
    float lightPdf, pdfPos, pdfDir ;

    float3 pos = Sample_Le(r0, r1, r2, r3, normal, lightDir, throughput, lightPdf, pdfPos, pdfDir);

    float3 beta = throughput * dot(normal, lightDir) / (lightPdf * pdfPos * pdfDir);

    float vertex_l_probability = lightPdf * pdfPos;
    float dL = NKK / vertex_l_probability;
    float vertex_l_solidangle = pdfDir;

    pathStateData[jobIndex * 3] = make_float4(throughput, dL);
    pathStateData[jobIndex * 3 + 1] = make_float4(beta, vertex_l_probability);
    pathStateData[jobIndex * 3 + 2] = make_float4(pos, vertex_l_solidangle);
}

//  +-----------------------------------------------------------------------------+
//  |  constructionLightPos                                                            |
//  |  Entry point for the persistent constructionLightPos kernel.               LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void constructionLightPos( int smcount, float NKK, uint* constructLightBuffer, float4* pathStateData, const uint R0, const uint* blueNoise, const int4 screenParams)
{
	const dim3 gridDim( NEXTMULTIPLEOF(smcount, 256 ) / 256, 1 ), blockDim( 256, 1 );
    constructionLightPosKernel << < gridDim.x, 256 >> > (smcount, NKK, constructLightBuffer, pathStateData, R0, blueNoise, screenParams);
}

// EOF
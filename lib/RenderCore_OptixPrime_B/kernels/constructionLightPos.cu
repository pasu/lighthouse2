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
void constructionLightPosKernel(int smcount, float NKK,uint* constructLightBuffer, 
    BiPathState* pathStateData, const uint R0, const uint* blueNoise, const int4 screenParams,
    Ray4* randomWalkRays)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= counters->activePaths) return;

    int jobIndex = constructLightBuffer[gid];

    const int scrhsize = screenParams.x & 0xffff;
    const int scrvsize = screenParams.x >> 16;
    const uint x = jobIndex % scrhsize;
    uint y = jobIndex / scrhsize;

    uint path_s_t_type_pass = pathStateData[jobIndex].pathInfo.w;

    uint s = 0;
    uint t = 1;
    uint type = 0;
    uint sampleIdx = path_s_t_type_pass & 255;

    float r0,r1,r2,r3;

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

    float light_p = lightPdf * pdfPos;
    float dL = NKK / light_p;
    float light_pdf_solid = pdfDir;

    const uint randomWalkRayIdx = atomicAdd(&counters->randomWalkRays, 1);
    randomWalkRays[randomWalkRayIdx].O4 = make_float4(pos, EPSILON);
    randomWalkRays[randomWalkRayIdx].D4 = make_float4(lightDir, 1e34f);

    pathStateData[jobIndex].data0 = make_float4(throughput, dL);
    pathStateData[jobIndex].data1 = make_float4(beta, light_p);
    pathStateData[jobIndex].data2 = make_float4(pos, light_pdf_solid);
    pathStateData[jobIndex].data3 = make_float4(lightDir, __int_as_float(randomWalkRayIdx));
    pathStateData[jobIndex].light_normal = make_float4(normal, 0.0f);

    sampleIdx++;
    path_s_t_type_pass = (s << 24) + (t<<16) + (type<<8) + sampleIdx;

    pathStateData[jobIndex].pathInfo.w = path_s_t_type_pass;
}

//  +-----------------------------------------------------------------------------+
//  |  constructionLightPos                                                            |
//  |  Entry point for the persistent constructionLightPos kernel.               LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void constructionLightPos( int smcount, float NKK, uint* constructLightBuffer, 
    BiPathState* pathStateData, const uint R0, const uint* blueNoise, const int4 screenParams,
    Ray4* randomWalkRays)
{
	const dim3 gridDim( NEXTMULTIPLEOF(smcount, 256 ) / 256, 1 ), blockDim( 256, 1 );
    constructionLightPosKernel << < gridDim.x, 256 >> > (smcount, NKK, constructLightBuffer, pathStateData, R0, blueNoise, screenParams, randomWalkRays);
}

// EOF
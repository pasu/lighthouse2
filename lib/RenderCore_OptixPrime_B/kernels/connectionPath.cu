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
    const Intersection* randomWalkHitBuffer, uint* visibilityHitBuffer,
    const float aperture, const float imgPlaneSize, const float3 forward,
    const float focalDistance, const float3 p1, const float3 right, const float3 up,
    const float spreadAngle, float4* accumulatorOnePass, uint* constructLightBuffer)
{
    int jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (jobIndex >= smcount) return;

    uint path_s_t_type_pass = pathStateData[jobIndex].pathInfo.w;

    uint pass, type, t, s;
    getPathInfo(path_s_t_type_pass,pass,s,t,type);

    float3 L = make_float3(0.0f);
    float misWeight = 0.0f;

    const uint occluded = visibilityHitBuffer[jobIndex >> 5] & (1 << (jobIndex & 31));

    if (type == 1)
    {
        float4 hitData = pathStateData[jobIndex].currentEye_hitData;
        float3 dir = make_float3(pathStateData[jobIndex].pre_eye_dir);

        float3 throughput = make_float3(pathStateData[jobIndex].data4);
        float3 eye_pos= make_float3(pathStateData[jobIndex].data6);
        float3 pre_pos = eye_pos - dir * HIT_T;
        float dE = pathStateData[jobIndex].data4.w;

        const int prim = __float_as_int(hitData.z);
        const int primIdx = prim == -1 ? prim : (prim & 0xfffff);

        const CoreTri4* instanceTriangles = (const CoreTri4*)instanceDescriptors[INSTANCEIDX].triangles;

        ShadingData shadingData;
        float3 N, iN, fN, T;

        const float coneWidth = spreadAngle * HIT_T;
        GetShadingData(dir, HIT_U, HIT_V, coneWidth, instanceTriangles[primIdx], INSTANCEIDX, shadingData, N, iN, fN, T);

        // implicit path s == k
        if (shadingData.IsEmissive() /* r, g or b exceeds 1 */)
        {
            L = throughput * shadingData.color;

            const CoreTri& tri = (const CoreTri&)instanceTriangles[primIdx];
            const float pickProb = LightPickProb(tri.ltriIdx, pre_pos, dir, eye_pos);
            const float pdfPos = 1.0f / tri.area;

            const float p_rev = pickProb * pdfPos;

            misWeight = 1.0f / (dE * p_rev + NKK);
        }
        else if (!occluded)
        {
            if (t == 1)
            {
                float3 light_pos = make_float3(pathStateData[jobIndex].data2);
                float3 light2eye = light_pos - eye_pos;
                float length_l2e = length(light2eye);
                light2eye /= length_l2e;

                float bsdfPdf;
                const float3 sampledBSDF = EvaluateBSDF(shadingData, fN, T, dir * -1.0f,light2eye, bsdfPdf);

                float3 light_throughput = make_float3(pathStateData[jobIndex].data0);
                float light_pdf = pathStateData[jobIndex].data1.w;
                
                float3 light_normal = make_float3(pathStateData[jobIndex].light_normal);
                float light_cosTheta = fabs(dot(light2eye * -1.0f, light_normal));

                // area to solid angle
                light_pdf *= length_l2e * length_l2e / light_cosTheta;

                float cosTheta = fabs(dot(fN, light2eye));

                L = throughput * sampledBSDF * light_throughput * (1.0f / light_pdf)  * cosTheta;                

                float3 eye_normal = make_float3(pathStateData[jobIndex].eye_normal);
                float eye_cosTheta = fabs(dot(light2eye, eye_normal));

                float p_forward = bsdfPdf * light_cosTheta / (length_l2e * length_l2e);
                float p_rev = light_cosTheta * INVPI * eye_cosTheta / (length_l2e * length_l2e);

                float dE = pathStateData[jobIndex].data4.w;
                float dL = pathStateData[jobIndex].data0.w;

                misWeight = 1.0 / (dE * p_rev + 1 + dL * p_forward);

                if (bsdfPdf < EPSILON || isnan(bsdfPdf))
                {
                    misWeight = 0.0f;
                }
            }
            else
            {
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
                int idx = (primIdx_light >> 20);

                const CoreTri4* instanceTriangles = (const CoreTri4*)instanceDescriptors[idx].triangles;

                ShadingData shadingData_light;
                float3 N_light, iN_light, fN_light, T_light;

                const float coneWidth = spreadAngle * HIT_T;
                GetShadingData(dir_light, HIT_U, HIT_V, coneWidth, instanceTriangles[primIdx_light],
                    idx, shadingData_light, N_light, iN_light, fN_light, T_light);

                float light_bsdfPdf;
                const float3 sampledBSDF_t = EvaluateBSDF(shadingData_light, fN_light, T_light, 
                    dir_light * -1.0f, light2eye * -1.0f, light_bsdfPdf);

                float3 throughput_light = make_float3(pathStateData[jobIndex].data0);

                // fabs keep safety
                float cosTheta_eye = fabs(dot(fN, light2eye));
                float cosTheta_light = fabs(dot(fN_light, light2eye* -1.0f));
                float G = cosTheta_eye * cosTheta_light / (length_l2e * length_l2e);

                L = throughput * sampledBSDF_s * sampledBSDF_t * throughput_light * G;

                float p_forward = eye_bsdfPdf * cosTheta_light / (length_l2e * length_l2e);
                float p_rev = light_bsdfPdf * cosTheta_eye / (length_l2e * length_l2e);

                float dE = pathStateData[jobIndex].data4.w;
                float dL = pathStateData[jobIndex].data0.w;

                misWeight = 1.0 / (dE * p_rev + 1 + dL * p_forward);

                if (eye_bsdfPdf < EPSILON || isnan(eye_bsdfPdf) 
                    || light_bsdfPdf < EPSILON || isnan(light_bsdfPdf))
                {
                    misWeight = 0.0f;
                }
            }
        }
    }
    else if (type == 2 && !occluded)
    {
        float3 light_pos = make_float3(pathStateData[jobIndex].data2);
        float3 eye_pos = make_float3(pathStateData[jobIndex].data6);

        float3 light2eye = eye_pos - light_pos;
        float length_l2e = length(light2eye);
        light2eye /= length_l2e;

        float3 throughput_eye;
        float pdf_eye;
        Sample_Wi(aperture,imgPlaneSize,eye_pos,forward,light_pos,
            focalDistance, p1, right, up, throughput_eye, pdf_eye);

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

            float3 throughput = make_float3(pathStateData[jobIndex].data0);
            float cosTheta = fabs(dot(fN, light2eye));

            L = throughput * sampledBSDF * (throughput_eye / pdf_eye) * cosTheta;

            float eye_cosTheta = fabs(dot(forward, light2eye * -1.0f));
            float eye_pdf_solid = 1.0f / (imgPlaneSize * eye_cosTheta * eye_cosTheta * eye_cosTheta);
            float p_forward = eye_pdf_solid * cosTheta / (length_l2e * length_l2e);

            float dL = pathStateData[jobIndex].data0.w;

            misWeight = 1.0f / (1 + dL * p_forward);

            if (bsdfPdf < EPSILON || isnan(bsdfPdf))
            {
                misWeight = 0.0f;
            }
        }
    }
    //misWeight = 1.0f;
    accumulatorOnePass[jobIndex] += make_float4((L*misWeight), 0.0f);
    
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
    float light_pdf = pathStateData[jobIndex].data2.w;

    if (light_pdf < EPSILON || isnan(light_pdf))
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
    const uint MAX__LENGTH_L = 3;

    if (eye_hit != -1 && s < MAX__LENGTH_E)
    {
        type = 1;
    }
    else if (light_hit != -1 && t < MAX__LENGTH_L)
    {
        type = 2;
    }
    else
    {
        const uint constructLight = atomicAdd(&counters->activePaths, 1);
        constructLightBuffer[constructLight] = jobIndex;
    }

    if (eye_hit == -1 && type != 2)
    {
        float3 hit_dir = make_float3(pathStateData[jobIndex].data7);
        float3 background = make_float3(SampleSkydome(hit_dir, s+1));

        float pdf_solidangle = pathStateData[jobIndex].data6.w;
        // hit miss : beta 
        float3 beta = make_float3(pathStateData[jobIndex].data5);
        float3 contribution = beta * background;

        CLAMPINTENSITY; // limit magnitude of thoughput vector to combat fireflies
        FIXNAN_FLOAT3(contribution);

        float dE = pathStateData[jobIndex].data4.w;
        misWeight = 1.0f / (dE * (1.0f / (scene_area)) + NKK);

        accumulatorOnePass[jobIndex] += make_float4((contribution * misWeight),0.0f);
    }

    //accumulatorOnePass[jobIndex] = make_float4(1.0, 0.0, 0.0, 1.0);
    //type = 0;
    path_s_t_type_pass = (s << 27) + (t << 22) + (type << 19) + pass;
    pathStateData[jobIndex].pathInfo.w = path_s_t_type_pass;

//    const uint constructLight = atomicAdd(&counters->activePaths, 1);
//    constructLightBuffer[constructLight] = jobIndex;

//    accumulatorOnePass[jobIndex] = make_float4(1.0, 0.0, 0.0, 1.0);
}

//  +-----------------------------------------------------------------------------+
//  |  constructionLightPos                                                            |
//  |  Entry point for the persistent constructionLightPos kernel.               LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void connectionPath(int smcount, float NKK, float scene_area, BiPathState* pathStateBuffer,
    const Intersection* randomWalkHitBuffer, uint* visibilityHitBuffer,
    const float aperture, const float imgPlaneSize, const float3 forward, 
    const float focalDistance, const float3 p1, const float3 right, const float3 up,
    const float spreadAngle, float4* accumulatorOnePass, uint* constructLightBuffer)
{
	const dim3 gridDim( NEXTMULTIPLEOF(smcount, 256 ) / 256, 1 ), blockDim( 256, 1 );
    connectionPathKernel << < gridDim.x, 256 >> > (smcount, NKK, scene_area, pathStateBuffer,
        randomWalkHitBuffer,visibilityHitBuffer, aperture, imgPlaneSize,
        forward, focalDistance, p1, right, up, spreadAngle, accumulatorOnePass, constructLightBuffer);
}

// EOF
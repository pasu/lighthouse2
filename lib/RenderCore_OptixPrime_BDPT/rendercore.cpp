/* rendercore.cpp - Copyright 2019 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   Implementation of the Optix Prime rendercore. This is a wavefront
   / streaming path tracer: CUDA code in camera.cu is used to
   generate a primary ray buffer, which is then traced by Optix. The
   resulting hitpoints are procressed using another CUDA kernel (in
   pathtracer.cu), which in turn generates extension rays and shadow
   rays. Path contributions are accumulated in an accumulator and
   finalized using code in finalize.cu.
*/

#include "core_settings.h"

namespace lh2core
{

// forward declaration of cuda code
const surfaceReference* renderTargetRef();
void InitCountersForExtend( int pathCount );
void InitContributions();

// setters / getters
void SetInstanceDescriptors( CoreInstanceDesc* p );
void SetMaterialList( CoreMaterial* p );
void SetAreaLights( CoreLightTri* p );
void SetPointLights( CorePointLight* p );
void SetSpotLights( CoreSpotLight* p );
void SetDirectionalLights( CoreDirectionalLight* p );
void SetLightCounts( int area, int point, int spot, int directional );
void SetARGB32Pixels( uint* p );
void SetARGB128Pixels( float4* p );
void SetNRM32Pixels( uint* p );
void SetSkyPixels( float3* p );
void SetSkySize( int w, int h );
void SetDebugData( float4* p );
void SetGeometryEpsilon( float e );
void SetClampValue( float c );
void SetCounters( Counters* p );

// BDPT
////////////////////////////////////////
void InitIndexForConstructionLight(int pathCount, uint* constructLightBuffer);
void constructionLightPos(int pathCount, float NKK, uint* constructLightBuffer, 
    BiPathState* pathStateBuffer, const uint R0, const uint* blueNoise, const int4 screenParams,
    Ray4* randomWalkRays, float4* accumulatorOnePass, float4* accumulator, 
    const int probePixelIdx, uint* constructEyeBuffer);
void constructionEyePos(int pathCount, uint* constructEyeBuffer,
    BiPathState* pathStateBuffer, Ray4* visibilityRays, Ray4* randomWalkRays,
    const uint R0, const float aperture, const float imgPlaneSize, const float3 camPos,
    const float3 right, const float3 up, const float3 forward, const float3 p1,
    const int4 screenParams);
void extendEyePath(int pathCount, BiPathState* pathStateBuffer,
    Ray4* visibilityRays, Ray4* randomWalkRays,
    const uint R0, const uint* blueNoise, const float spreadAngle,
    const int4 screenParams, const int probePixelIdx, uint* eyePathBuffer,
    uint* contributionBuffer_Emissive,uint* contributionBuffer_Explicit, 
    uint* contributionBuffer_Connection, float4* accumulatorOnePass, float NKK);
void extendLightPath(int smcount, BiPathState* pathStateBuffer,
    Ray4* visibilityRays, Ray4* randomWalkRays, const uint R0, const uint* blueNoise,
    const float3 camPos, const float spreadAngle,const int4 screenParams, 
    uint* lightPathBuffer, uint* contributionBuffer_Photon,
    const float aperture, const float imgPlaneSize,
    const float3 forward, const float focalDistance, const float3 p1,
    const float3 right, const float3 up);
void extendPath(int pathCount, BiPathState* pathStateBuffer,
    Ray4* visibilityRays, Ray4* randomWalkRays,
    const uint R0, const uint* blueNoise, const float lensSize, const float imgPlaneSize, const float3 camPos,
    const float3 right, const float3 up, const float3 forward, const float3 p1, const float spreadAngle,
    const int4 screenParams, const int probePixelIdx);

void connectionPath_Emissive(int smcount, float NKK, BiPathState* pathStateData,
    const float spreadAngle, float4* accumulatorOnePass,
    const int4 screenParams,
    uint* contributionBuffer_Emissive);

void connectionPath_Explicit(int smcount, BiPathState* pathStateData,
    uint* visibilityHitBuffer, const float spreadAngle,
    float4* accumulatorOnePass,
    const int4 screenParams, uint* contributionBuffer_Explicit, 
    const int probePixelIdx);

void connectionPath_Connection(int smcount, BiPathState* pathStateData,
    uint* visibilityHitBuffer, const float spreadAngle,
    float4* accumulatorOnePass,
    const int4 screenParams, uint* contributionBuffer_Connection);

void connectionPath_Photon(int smcount, BiPathState* pathStateData,
    uint* visibilityHitBuffer, const float aperture, const float imgPlaneSize,
    const float3 forward, const float focalDistance, const float3 p1,
    const float3 right, const float3 up, const float spreadAngle,
    float4* accumulatorOnePass, const int4 screenParams,
    const float3 camPos,
    uint* contributionBuffer_Photon, const int probePixelIdx);

void connectionPath(int smcount, float NKK, float scene_area, BiPathState* pathStateData,
    const Intersection* randomWalkHitBuffer,
    float4* accumulatorOnePass, uint* constructLightBuffer,
    const int4 screenParams,
    uint* constructEyeBuffer, uint* eyePathBuffer, uint* lightPathBuffer);

void finalizeRender_BDPT(const float4* accumulator, 
    const int w, const int h, const float brightness, const float contrast);
//////////////////////////////////////////

} // namespace lh2core

using namespace lh2core;

RTPcontext RenderCore::context = 0;

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::GetScreenParams                                                |
//  |  Helper function - fills an int4 with values related to screen size.  LH2'19|
//  +-----------------------------------------------------------------------------+
int4 RenderCore::GetScreenParams()
{
	float e = 0.0001f; // RenderSettings::geoEpsilon;
	return make_int4( scrwidth + (scrheight << 16),					// .x : SCRHSIZE, SCRVSIZE
		scrspp + (1 /* RenderSettings::pathDepth */ << 8),			// .y : SPP, MAXDEPTH
		scrwidth * scrheight * scrspp,								// .z : PIXELCOUNT
		*((int*)&e) );												// .w : RenderSettings::geoEpsilon
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetProbePos                                                    |
//  |  Set the pixel for which the triid will be captured.                  LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetProbePos( int2 pos )
{
	probePos = pos; // triangle id for this pixel will be stored in coreStats
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Init                                                           |
//  |  CUDA / Optix / RenderCore initialization.                            LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Init()
{
	// select the fastest device
	uint device = CUDATools::FastestDevice();
	cudaSetDevice( device );
	cudaDeviceProp properties;
	cudaGetDeviceProperties( &properties, device );
	coreStats.SMcount = SMcount = properties.multiProcessorCount;
	coreStats.ccMajor = properties.major;
	coreStats.ccMinor = properties.minor;
	computeCapability = coreStats.ccMajor * 10 + coreStats.ccMinor;
	coreStats.VRAM = (uint)(properties.totalGlobalMem >> 20);
	coreStats.deviceName = new char[strlen( properties.name ) + 1];
	memcpy( coreStats.deviceName, properties.name, strlen( properties.name ) + 1 );
	printf( "running on GPU: %s (%i SMs, %iGB VRAM)\n", coreStats.deviceName, coreStats.SMcount, (int)(coreStats.VRAM >> 10) );
	// setup OptiX Prime
	CHK_PRIME( rtpContextCreate( RTP_CONTEXT_TYPE_CUDA, &context ) );
	const char* versionString;
	CHK_PRIME( rtpGetVersionString( &versionString ) );
	printf( "%s\n", versionString );
	CHK_PRIME( rtpContextSetCudaDeviceNumbers( context, 1, &device ) );
	// prepare the top-level 'model' node; instances will be added to this.
	topLevel = new RTPmodel();
	CHK_PRIME( rtpModelCreate( context, topLevel ) );
	// prepare counters for persistent threads
	counterBuffer = new CoreBuffer<Counters>( 16, ON_DEVICE );
	SetCounters( counterBuffer->DevPtr() );
	// render settings
	SetClampValue( 10.0f );
	// prepare the bluenoise data
	const uchar* data8 = (const uchar*)sob256_64; // tables are 8 bit per entry
	uint* data32 = new uint[65536 * 5]; // we want a full uint per entry
	for( int i = 0; i < 65536; i++ ) data32[i] = data8[i]; // convert
	data8 = (uchar*)scr256_64;
	for( int i = 0; i < (128 * 128 * 8); i++ ) data32[i + 65536] = data8[i];
	data8 = (uchar*)rnk256_64;
	for( int i = 0; i < (128 * 128 * 8); i++ ) data32[i + 3 * 65536] = data8[i];
	blueNoise = new CoreBuffer<uint>( 65536 * 5, ON_DEVICE, data32 );
	delete data32;
	// allow CoreMeshes to access the core
	CoreMesh::renderCore = this;
	// timing events
	for( int i = 0; i < MAXPATHLENGTH; i++ )
	{
		cudaEventCreate( &shadeStart[i] );
		cudaEventCreate( &shadeEnd[i] );
	}
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetTarget                                                      |
//  |  Set the OpenGL texture that serves as the render target.             LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetTarget( GLTexture* target, const uint spp )
{
	// synchronize OpenGL viewport
	scrwidth = target->width;
	scrheight = target->height;
	scrspp = 1;
	renderTarget.SetTexture( target );
	bool firstFrame = (maxPixels == 0);
	// notify CUDA about the texture
	renderTarget.LinkToSurface( renderTargetRef() );
	// see if we need to reallocate our buffers
	bool reallocate = false;
	if (scrwidth * scrheight > maxPixels || spp != currentSPP)
	{
		maxPixels = scrwidth * scrheight;
		maxPixels += maxPixels >> 4; // reserve a bit extra to prevent frequent reallocs
		currentSPP = spp;
		reallocate = true;
	}
	if (reallocate)
	{
		// destroy previously created OptiX buffers
		if (!firstFrame)
		{
            //BDPT
            //////////////////////////
            rtpBufferDescDestroy(visibilityRaysDesc);
            rtpBufferDescDestroy(visibilityHitsDesc);
            rtpBufferDescDestroy(randomWalkRaysDesc);
            rtpBufferDescDestroy(randomWalkHitsDesc);
            /////////////////////////
		}
		// delete CoreBuffers
		delete accumulator;
        delete accumulatorOnePass;
        delete weightMeasureBuffer;
        // BDPT
        /////////////////////////////
        delete constructLightBuffer;
        delete constructEyeBuffer;
        
        delete eyePathBuffer;
        delete lightPathBuffer;

        delete contributionBuffer_Emissive;
        delete contributionBuffer_Explicit;
        delete contributionBuffer_Connection;
        delete contributionBuffer_Photon;

        delete pathDataBuffer;
        delete visibilityRayBuffer;
        delete visibilityHitBuffer;
        delete randomWalkRayBuffer;
        delete randomWalkHitBuffer;
        /////////////////////////////

 		accumulator = new CoreBuffer<float4>( maxPixels, ON_DEVICE );
        accumulatorOnePass = new CoreBuffer<float4>(maxPixels, ON_DEVICE);
        //weightMeasureBuffer = new CoreBuffer<float4>(maxPixels, ON_DEVICE);    
        // BDPT
        ///////////////////////////////////////////
        constructLightBuffer = new CoreBuffer<uint>( maxPixels * spp, ON_DEVICE );
        constructEyeBuffer = new CoreBuffer<uint>(maxPixels * spp, ON_DEVICE);

        eyePathBuffer = new CoreBuffer<uint>(maxPixels * spp, ON_DEVICE);
        lightPathBuffer = new CoreBuffer<uint>(maxPixels * spp, ON_DEVICE);

        contributionBuffer_Emissive = new CoreBuffer<uint>(maxPixels * spp, ON_DEVICE);
        contributionBuffer_Explicit = new CoreBuffer<uint>(maxPixels * spp, ON_DEVICE);
        contributionBuffer_Connection = new CoreBuffer<uint>(maxPixels * spp, ON_DEVICE);
        contributionBuffer_Photon = new CoreBuffer<uint>(maxPixels * spp, ON_DEVICE);

        pathDataBuffer = new CoreBuffer<BiPathState>(maxPixels * spp, ON_DEVICE);

        //photomapping = new CoreBuffer<float4>(maxPixels * spp, ON_DEVICE);
        //photomappingIdx = new CoreBuffer<uint>(maxPixels * spp, ON_DEVICE);
        
        visibilityRayBuffer = new CoreBuffer<Ray4>(maxPixels * spp, ON_DEVICE);
        CHK_PRIME(rtpBufferDescCreate(context, RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX, RTP_BUFFER_TYPE_CUDA_LINEAR, visibilityRayBuffer->DevPtr(), &visibilityRaysDesc));
        visibilityHitBuffer = new CoreBuffer<uint>((maxPixels * spp + 31) >> 5, ON_DEVICE);
        CHK_PRIME(rtpBufferDescCreate(context, RTP_BUFFER_FORMAT_HIT_BITMASK, RTP_BUFFER_TYPE_CUDA_LINEAR, visibilityHitBuffer->DevPtr(), &visibilityHitsDesc));
        randomWalkRayBuffer = new CoreBuffer<Ray4>(maxPixels * 2 * spp, ON_DEVICE);
        CHK_PRIME(rtpBufferDescCreate(context, RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX, RTP_BUFFER_TYPE_CUDA_LINEAR, randomWalkRayBuffer->DevPtr(), &randomWalkRaysDesc));
        randomWalkHitBuffer = new CoreBuffer<Intersection>(maxPixels * 2 * spp, ON_DEVICE);
        CHK_PRIME(rtpBufferDescCreate(context, RTP_BUFFER_FORMAT_HIT_T_TRIID_INSTID_U_V, RTP_BUFFER_TYPE_CUDA_LINEAR, randomWalkHitBuffer->DevPtr(), &randomWalkHitsDesc));
        //////////////////////////////////////////
		printf( "buffers resized for %i pixels @ %i samples.\n", maxPixels, spp );
	}
	// clear the accumulator
 	accumulator->Clear( ON_DEVICE );
    accumulatorOnePass->Clear(ON_DEVICE);
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetGeometry                                                    |
//  |  Set the geometry data for a model.                                   LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetGeometry( const int meshIdx, const float4* vertexData, const int vertexCount, const int triangleCount, const CoreTri* triangles, const uint* alphaFlags )
{
	// Note: for first-time setup, meshes are expected to be passed in sequential order.
	// This will result in new CoreMesh pointers being pushed into the meshes vector.
	// Subsequent mesh changes will be applied to existing CoreMeshes. This is deliberately
	// minimalistic; RenderSystem is responsible for a proper (fault-tolerant) interface.
	if (meshIdx >= meshes.size()) meshes.push_back( new CoreMesh() );
	meshes[meshIdx]->SetGeometry( vertexData, vertexCount, triangleCount, triangles, alphaFlags );
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetInstance                                                    |
//  |  Set instance details.                                                LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetInstance( const int instanceIdx, const int meshIdx, const mat4& matrix )
{
	// Note: for first-time setup, meshes are expected to be passed in sequential order.
	// This will result in new CoreInstance pointers being pushed into the instances vector.
	// Subsequent instance changes (typically: transforms) will be applied to existing CoreInstances.
	if (instanceIdx >= instances.size()) instances.push_back( new CoreInstance() );
	instances[instanceIdx]->mesh = meshIdx;
	instances[instanceIdx]->transform = matrix;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::UpdateToplevel                                                 |
//  |  After changing meshes, instances or instance transforms, we need to        |
//  |  rebuild the top-level structure.                                     LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::UpdateToplevel()
{
	// this creates the top-level BVH over the supplied models.
	RTPbufferdesc instancesBuffer, transformBuffer;
	vector<RTPmodel> modelList;
	vector<mat4> transformList;
	for (auto instance : instances)
	{
		modelList.push_back( meshes[instance->mesh]->model );
		transformList.push_back( instance->transform );
	}
	CHK_PRIME( rtpBufferDescCreate( context, RTP_BUFFER_FORMAT_INSTANCE_MODEL, RTP_BUFFER_TYPE_HOST, modelList.data(), &instancesBuffer ) );
	CHK_PRIME( rtpBufferDescCreate( context, RTP_BUFFER_FORMAT_TRANSFORM_FLOAT4x4, RTP_BUFFER_TYPE_HOST, transformList.data(), &transformBuffer ) );
	CHK_PRIME( rtpBufferDescSetRange( instancesBuffer, 0, instances.size() ) );
	CHK_PRIME( rtpBufferDescSetRange( transformBuffer, 0, instances.size() ) );
	CHK_PRIME( rtpModelSetInstances( *topLevel, instancesBuffer, transformBuffer ) );
	CHK_PRIME( rtpModelUpdate( *topLevel, RTP_MODEL_HINT_ASYNC /* blocking; try RTP_MODEL_HINT_ASYNC + rtpModelFinish for async version. */ ) );
	CHK_PRIME( rtpBufferDescDestroy( instancesBuffer ) /* no idea if this does anything relevant */ );
	CHK_PRIME( rtpBufferDescDestroy( transformBuffer ) /* no idea if this does anything relevant */ );
	instancesDirty = true; // sync instance list to device prior to next ray query
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetTextures                                                    |
//  |  Set the texture data.                                                LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetTextures( const CoreTexDesc* tex, const int textures )
{
	// copy the supplied array of texture descriptors
	delete texDescs; texDescs = 0;
	textureCount = textures;
	if (textureCount == 0) return; // scene has no textures
	texDescs = new CoreTexDesc[textureCount];
	memcpy( texDescs, tex, textureCount * sizeof( CoreTexDesc ) );
	// copy texels for each type to the device
	SyncStorageType( TexelStorage::ARGB32 );
	SyncStorageType( TexelStorage::ARGB128 );
	SyncStorageType( TexelStorage::NRM32 );
	// Notes: 
	// - the three types are copied from the original HostTexture pixel data (to which the
	//   descriptors point) straight to the GPU. There is no pixel storage on the host
	//   in the RenderCore.
	// - the types are copied one by one. Copying involves creating a temporary host-side
	//   buffer; doing this one by one allows us to delete host-side data for one type
	//   before allocating space for the next, thus reducing storage requirements.
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SyncStorageType                                                |
//  |  Copies texel data for one storage type (argb32, argb128 or nrm32) to the   |
//  |  device. Note that this data is obtained from the original HostTexture      |
//  |  texel arrays.                                                        LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SyncStorageType( const TexelStorage storage )
{
	uint texelTotal = 0;
	for (int i = 0; i < textureCount; i++) if (texDescs[i].storage == storage) texelTotal += texDescs[i].pixelCount;
	texelTotal = max( 16, texelTotal ); // OptiX does not tolerate empty buffers...
	// construct the continuous arrays
	switch (storage)
	{
	case TexelStorage::ARGB32:
		delete texel32Buffer;
		texel32Buffer = new CoreBuffer<uint>( texelTotal, ON_HOST | ON_DEVICE );
		SetARGB32Pixels( texel32Buffer->DevPtr() );
		coreStats.argb32TexelCount = texelTotal;
		break;
	case TexelStorage::ARGB128:
		delete texel128Buffer;
		SetARGB128Pixels( (texel128Buffer = new CoreBuffer<float4>( texelTotal, ON_HOST | ON_DEVICE ))->DevPtr() );
		coreStats.argb128TexelCount = texelTotal;
		break;
	case TexelStorage::NRM32:
		delete normal32Buffer;
		SetNRM32Pixels( (normal32Buffer = new CoreBuffer<uint>( texelTotal, ON_HOST | ON_DEVICE ))->DevPtr() );
		coreStats.nrm32TexelCount = texelTotal;
		break;
	}
	// copy texel data to arrays
	texelTotal = 0;
	for (int i = 0; i < textureCount; i++) if (texDescs[i].storage == storage)
	{
		void* destination = 0;
		switch (storage)
		{
		case TexelStorage::ARGB32:  destination = texel32Buffer->HostPtr() + texelTotal; break;
		case TexelStorage::ARGB128: destination = texel128Buffer->HostPtr() + texelTotal; break;
		case TexelStorage::NRM32:   destination = normal32Buffer->HostPtr() + texelTotal; break;
		}
		memcpy( destination, texDescs[i].idata, texDescs[i].pixelCount * sizeof( uint ) );
		texDescs[i].firstPixel = texelTotal;
		texelTotal += texDescs[i].pixelCount;
	}
	// move to device
	if (storage == TexelStorage::ARGB32) if (texel32Buffer) texel32Buffer->MoveToDevice();
	if (storage == TexelStorage::ARGB128) if (texel128Buffer) texel128Buffer->MoveToDevice();
	if (storage == TexelStorage::NRM32) if (normal32Buffer) normal32Buffer->MoveToDevice();
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetMaterials                                                   |
//  |  Set the material data.                                               LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetMaterials( CoreMaterial* mat, const CoreMaterialEx* matEx, const int materialCount )
{
	// Notes:
	// Call this after the textures have been set; CoreMaterials store the offset of each texture
	// in the continuous arrays; this data is valid only when textures are in sync.
	delete materialBuffer;
	delete hostMaterialBuffer;
	hostMaterialBuffer = new CoreMaterial[materialCount];
	memcpy( hostMaterialBuffer, mat, materialCount * sizeof( CoreMaterial ) );
	for (int i = 0; i < materialCount; i++)
	{
		CoreMaterial& m = hostMaterialBuffer[i];
		const CoreMaterialEx& e = matEx[i];
		if (e.texture[0] != -1) m.texaddr0 = texDescs[e.texture[0]].firstPixel;
		if (e.texture[1] != -1) m.texaddr1 = texDescs[e.texture[1]].firstPixel;
		if (e.texture[2] != -1) m.texaddr2 = texDescs[e.texture[2]].firstPixel;
		if (e.texture[3] != -1) m.nmapaddr0 = texDescs[e.texture[3]].firstPixel;
		if (e.texture[4] != -1) m.nmapaddr1 = texDescs[e.texture[4]].firstPixel;
		if (e.texture[5] != -1) m.nmapaddr2 = texDescs[e.texture[5]].firstPixel;
		if (e.texture[6] != -1) m.smapaddr = texDescs[e.texture[6]].firstPixel;
		if (e.texture[7] != -1) m.rmapaddr = texDescs[e.texture[7]].firstPixel;
		// if (e.texture[ 8] != -1) m.texaddr0 = texDescs[e.texture[ 8]].firstPixel; second roughness map is not used
		if (e.texture[9] != -1) m.cmapaddr = texDescs[e.texture[9]].firstPixel;
		if (e.texture[10] != -1) m.amapaddr = texDescs[e.texture[10]].firstPixel;
	}
	materialBuffer = new CoreBuffer<CoreMaterial>( materialCount, ON_DEVICE | ON_HOST /* on_host: for alpha mapped tris */, hostMaterialBuffer );
	SetMaterialList( materialBuffer->DevPtr() );
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetLights                                                      |
//  |  Set the light data.                                                  LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetLights( const CoreLightTri* areaLights, const int areaLightCount,
	const CorePointLight* pointLights, const int pointLightCount,
	const CoreSpotLight* spotLights, const int spotLightCount,
	const CoreDirectionalLight* directionalLights, const int directionalLightCount )
{
	delete areaLightBuffer;
	delete pointLightBuffer;
	delete spotLightBuffer;
	delete directionalLightBuffer;
	SetAreaLights( (areaLightBuffer = new CoreBuffer<CoreLightTri>( areaLightCount, ON_DEVICE, areaLights ))->DevPtr() );
	SetPointLights( (pointLightBuffer = new CoreBuffer<CorePointLight>( pointLightCount, ON_DEVICE, pointLights ))->DevPtr() );
	SetSpotLights( (spotLightBuffer = new CoreBuffer<CoreSpotLight>( spotLightCount, ON_DEVICE, spotLights ))->DevPtr() );
	SetDirectionalLights( (directionalLightBuffer = new CoreBuffer<CoreDirectionalLight>( directionalLightCount, ON_DEVICE, directionalLights ))->DevPtr() );
	SetLightCounts( areaLightCount, pointLightCount, spotLightCount, directionalLightCount );
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetSkyData                                                     |
//  |  Set the sky dome data.                                               LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetSkyData( const float3* pixels, const uint width, const uint height )
{
	delete skyPixelBuffer;
	skyPixelBuffer = new CoreBuffer<float3>( width * height, ON_DEVICE, pixels );
	SetSkyPixels( skyPixelBuffer->DevPtr() );
	SetSkySize( width, height );
	skywidth = width;
	skyheight = height;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Setting                                                        |
//  |  Modify a render setting.                                             LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Setting( const char* name, const float value )
{
	if (!strcmp( name, "epsilon" ))
	{
		if (vars.geometryEpsilon != value)
		{
			vars.geometryEpsilon = value;
			SetGeometryEpsilon( value );
		}
	}
	else if (!strcmp( name, "clampValue" ))
	{
		if (vars.clampValue != value)
		{
			vars.clampValue = value;
			SetClampValue( value );
		}
	}
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Render                                                         |
//  |  Produce one image.                                                   LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Render( const ViewPyramid& view, const Convergence converge, const float brightness, const float contrast )
{
	// wait for OpenGL
	glFinish();
	Timer timer;
	// clean accumulator, if requested
	if (converge == Restart)
	{
		accumulator->Clear( ON_DEVICE );
        accumulatorOnePass->Clear(ON_DEVICE);
        pathDataBuffer->Clear(ON_DEVICE);

        InitCountersForExtend(scrwidth * scrheight * scrspp);
        InitIndexForConstructionLight(scrwidth * scrheight * scrspp, constructLightBuffer->DevPtr());

		camRNGseed = 0x12345678; // same seed means same noise.
	}

    // BDPT
    ///////////////////////////////////
    static bool bInit = false;
    static float NKK = 4.1;
    if (!bInit)
    {
        InitCountersForExtend(scrwidth * scrheight * scrspp);
        InitIndexForConstructionLight(scrwidth * scrheight * scrspp, constructLightBuffer->DevPtr());
        bInit = true;
    }    
    //////////////////////////////////////
	// update instance descriptor array on device
	// Note: we are not using the built-in OptiX instance system for shading. Instead,
	// we figure out which triangle we hit, and to what instance it belongs; from there,
	// we handle normal management and material acquisition in custom code.
	if (instancesDirty)
	{
		// prepare CoreInstanceDesc array. For any sane number of instances this should
		// be efficient while yielding supreme flexibility.
		vector<CoreInstanceDesc> instDescArray;
		for (auto instance : instances)
		{
			CoreInstanceDesc id;
			id.triangles = meshes[instance->mesh]->triangles->DevPtr();
			mat4 T = instance->transform.Inverted();
			id.invTransform = *(float4x4*)&T;
			instDescArray.push_back( id );
		}
		if (instDescBuffer == 0 || instDescBuffer->GetSize() < (int)instances.size())
		{
			delete instDescBuffer;
			// size of instance list changed beyond capacity.
			// Allocate a new buffer, with some slack, to prevent excessive reallocs.
			instDescBuffer = new CoreBuffer<CoreInstanceDesc>( instances.size() * 2, ON_HOST | ON_DEVICE );
			SetInstanceDescriptors( instDescBuffer->DevPtr() );
		}
		memcpy( instDescBuffer->HostPtr(), instDescArray.data(), instDescArray.size() * sizeof( CoreInstanceDesc ) );
		instDescBuffer->CopyToDevice();
		// instancesDirty = false;
	}

    uint pathCount = scrwidth * scrheight * scrspp;
    uint lightCount = pathCount;
    float3 right = view.p2 - view.p1, up = view.p3 - view.p1;
    float3 forward = cross(right, up);
	// render image
    RTPquery queryVisibility,queryRandomWalk;
    CHK_PRIME(rtpQueryCreate(*topLevel, RTP_QUERY_TYPE_CLOSEST, &queryVisibility));
    CHK_PRIME(rtpQueryCreate(*topLevel, RTP_QUERY_TYPE_CLOSEST, &queryRandomWalk));

    for (int pathLength = 1; pathLength <= 8; pathLength++)
    {
        //             constructLightBuffer->CopyToHost();
        //             uint* index = constructLightBuffer->HostPtr();
        // 
        //             accumulatorOnePass->CopyToHost();
        //             float4* color = accumulatorOnePass->HostPtr();

        constructionLightPos(lightCount, NKK,
            constructLightBuffer->DevPtr(), pathDataBuffer->DevPtr(),
            RandomUInt(camRNGseed), blueNoise->DevPtr(), GetScreenParams(),
            randomWalkRayBuffer->DevPtr(), accumulatorOnePass->DevPtr(), 
            accumulator->DevPtr(), 
            probePos.x + scrwidth * probePos.y,constructEyeBuffer->DevPtr());
        
//         pathDataBuffer->CopyToHost();
//         BiPathState* state = pathDataBuffer->HostPtr();

        constructionEyePos(lightCount, constructEyeBuffer->DevPtr(),
            pathDataBuffer->DevPtr(), visibilityRayBuffer->DevPtr(),
            randomWalkRayBuffer->DevPtr(), RandomUInt(camRNGseed),
            view.aperture, view.imagePlane, view.pos, right, up, forward, view.p1,
            GetScreenParams());
        counterBuffer->CopyToHost();

        extendEyePath(pathCount, pathDataBuffer->DevPtr(),
            visibilityRayBuffer->DevPtr(), randomWalkRayBuffer->DevPtr(),
            RandomUInt(camRNGseed), blueNoise->DevPtr(), view.spreadAngle, GetScreenParams(),
            probePos.x + scrwidth * probePos.y, eyePathBuffer->DevPtr(),
            contributionBuffer_Emissive->DevPtr(),contributionBuffer_Explicit->DevPtr(),
            contributionBuffer_Connection->DevPtr(),
            accumulatorOnePass->DevPtr(), NKK);        

        extendLightPath(pathCount, pathDataBuffer->DevPtr(),
            visibilityRayBuffer->DevPtr(), randomWalkRayBuffer->DevPtr(),
            RandomUInt(camRNGseed), blueNoise->DevPtr(), view.pos,
            view.spreadAngle, GetScreenParams(), lightPathBuffer->DevPtr(),
            contributionBuffer_Photon->DevPtr(), 
            view.aperture, view.imagePlane, forward,
            view.focalDistance, view.p1, right, up);
        
        /*
        extendPath(pathCount, pathDataBuffer->DevPtr(),
            visibilityRayBuffer->DevPtr(), randomWalkRayBuffer->DevPtr(),
            RandomUInt(camRNGseed), blueNoise->DevPtr(),
            view.aperture, view.imagePlane, view.pos, right, up, forward, view.p1, view.spreadAngle, GetScreenParams(),
            probePos.x + scrwidth * probePos.y);
        */
        

        counterBuffer->CopyToHost();
        Counters& counters = counterBuffer->HostPtr()[0];

        CHK_PRIME(rtpBufferDescSetRange(visibilityRaysDesc, 0, pathCount));
        CHK_PRIME(rtpBufferDescSetRange(visibilityHitsDesc, 0, pathCount));
        CHK_PRIME(rtpQuerySetRays(queryVisibility, visibilityRaysDesc));
        CHK_PRIME(rtpQuerySetHits(queryVisibility, visibilityHitsDesc));
        CHK_PRIME(rtpQueryExecute(queryVisibility, RTP_QUERY_HINT_NONE));

        CHK_PRIME(rtpBufferDescSetRange(randomWalkRaysDesc, 0, counters.randomWalkRays));
        CHK_PRIME(rtpBufferDescSetRange(randomWalkHitsDesc, 0, counters.randomWalkRays));
        CHK_PRIME(rtpQuerySetRays(queryRandomWalk, randomWalkRaysDesc));
        CHK_PRIME(rtpQuerySetHits(queryRandomWalk, randomWalkHitsDesc));
        CHK_PRIME(rtpQueryExecute(queryRandomWalk, RTP_QUERY_HINT_NONE));
        /*
        connectionPath_Emissive(pathCount, NKK,pathDataBuffer->DevPtr(),
            view.spreadAngle,accumulatorOnePass->DevPtr(),
            GetScreenParams(),contributionBuffer_Emissive->DevPtr());
        *//**/
        connectionPath_Explicit(pathCount, pathDataBuffer->DevPtr(),
            visibilityHitBuffer->DevPtr(), view.spreadAngle, 
            accumulatorOnePass->DevPtr(), 
            GetScreenParams(),contributionBuffer_Explicit->DevPtr(),
            probePos.x + scrwidth * probePos.y);
        
        
        connectionPath_Connection(pathCount, pathDataBuffer->DevPtr(),
            visibilityHitBuffer->DevPtr(), view.spreadAngle,
            accumulatorOnePass->DevPtr(), 
            GetScreenParams(), contributionBuffer_Connection->DevPtr());
        
        connectionPath_Photon(pathCount, pathDataBuffer->DevPtr(),
            visibilityHitBuffer->DevPtr(), view.aperture, view.imagePlane, forward,
            view.focalDistance, view.p1, right, up,
            view.spreadAngle, accumulatorOnePass->DevPtr(), GetScreenParams(), 
            view.pos,
            contributionBuffer_Photon->DevPtr(),
            probePos.x + scrwidth * probePos.y);
        
        InitCountersForExtend(0);

        float scene_area = 5989.0f;
        connectionPath(pathCount, NKK, scene_area, pathDataBuffer->DevPtr(), randomWalkHitBuffer->DevPtr(),
            accumulatorOnePass->DevPtr(), constructLightBuffer->DevPtr(),
            GetScreenParams(), 
            constructEyeBuffer->DevPtr(),eyePathBuffer->DevPtr(),lightPathBuffer->DevPtr());

        counterBuffer->CopyToHost();
        counters = counterBuffer->HostPtr()[0];

        coreStats.probedInstid = counters.probedInstid;
        coreStats.probedTriid = counters.probedTriid;
        coreStats.probedDist = counters.probedDist;

        //InitContributions();
    }

    renderTarget.BindSurface();
    finalizeRender_BDPT(accumulator->DevPtr(), scrwidth, scrheight, brightness, contrast);
    renderTarget.UnbindSurface();

    CHK_PRIME(rtpQueryDestroy(queryVisibility));
    CHK_PRIME(rtpQueryDestroy(queryRandomWalk));    
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Shutdown                                                       |
//  |  Free all resources.                                                  LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Shutdown()
{
	// delete ray buffers
    delete weightMeasureBuffer;
    delete constructLightBuffer;
    delete constructEyeBuffer;

    delete eyePathBuffer;
    delete lightPathBuffer;

    delete contributionBuffer_Emissive;
    delete contributionBuffer_Explicit;
    delete contributionBuffer_Connection;
    delete contributionBuffer_Photon;

    delete pathDataBuffer;
    delete visibilityRayBuffer;
    delete visibilityHitBuffer;
    delete randomWalkRayBuffer;
    delete randomWalkHitBuffer;
    delete photomapping;
    delete photomappingIdx;
	// delete internal data
	delete accumulator;
    delete accumulatorOnePass;
	delete counterBuffer;
	delete texDescs;
	delete texel32Buffer;
	delete texel128Buffer;
	delete normal32Buffer;
	delete materialBuffer;
	delete hostMaterialBuffer;
	delete skyPixelBuffer;
	delete instDescBuffer;
	// delete light data
	delete areaLightBuffer;
	delete pointLightBuffer;
	delete spotLightBuffer;
	delete directionalLightBuffer;
	// delete core scene representation
	for (auto mesh : meshes) delete mesh;
	for (auto instance : instances) delete instance;
	delete topLevel;
	rtpBufferDescDestroy(visibilityRaysDesc );
	rtpBufferDescDestroy(visibilityHitsDesc );
	rtpBufferDescDestroy(randomWalkRaysDesc);
    rtpBufferDescDestroy(randomWalkHitsDesc);
	rtpContextDestroy( context );
}

// EOF
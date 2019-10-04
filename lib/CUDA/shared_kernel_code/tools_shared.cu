/* tools_shared.cu - Copyright 2019 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   THIS IS A SHARED FILE:
   used in RenderCore_OptixPrime, RenderCore_OptixRTX and RenderCore_RTXAO.
*/

#include "noerrors.h"

#define CHAR2FLT(a,s) (((float)(((a)>>s)&255))*(1.0f/255.0f))

struct ShadingData
{
	// This structure is filled for an intersection point. It will contain the spatially varying material properties.
	float3 color; int flags;
	float3 absorption; int matID;
	uint4 parameters;
	/* 16 uchars:   x: roughness, metallic, specTrans, specularTint;
					y: diffTrans, anisotropic, sheen, sheenTint;
					z: clearcoat, clearcoatGloss, scatterDistance, relativeIOR;
					w: flatness, ior, dummy1, dummy2. */
	__device__ int IsSpecular( const int layer ) const { return 0; /* for now. */ }
	__device__ bool IsEmissive() const { return color.x > 1.0f || color.y > 1.0f || color.z > 1.0f; }
#define METALLIC CHAR2FLT( shadingData.parameters.x, 0 )
#define SUBSURFACE CHAR2FLT( shadingData.parameters.x, 8 )
#define SPECULAR CHAR2FLT( shadingData.parameters.x, 16 )
#define ROUGHNESS (max( 0.001f, CHAR2FLT( shadingData.parameters.x, 24 ) ))
#define SPECTINT CHAR2FLT( shadingData.parameters.y, 0 )
#define ANISOTROPIC CHAR2FLT( shadingData.parameters.y, 8 )
#define SHEEN CHAR2FLT( shadingData.parameters.y, 16 )
#define SHEENTINT CHAR2FLT( shadingData.parameters.y, 24 )
#define CLEARCOAT CHAR2FLT( shadingData.parameters.z, 0 )
#define CLEARCOATGLOSS CHAR2FLT( shadingData.parameters.z, 8 )
#define TRANSMISSION CHAR2FLT( shadingData.parameters.z, 16 )
#define ETA CHAR2FLT( shadingData.parameters.z, 24 )
#define CUSTOM0 CHAR2FLT( shadingData.parameters.z, 24 )
#define CUSTOM1 CHAR2FLT( shadingData.parameters.w, 0 )
#define CUSTOM2 CHAR2FLT( shadingData.parameters.w, 8 )
#define CUSTOM3 CHAR2FLT( shadingData.parameters.w, 16 )
#define CUSTOM4 CHAR2FLT( shadingData.parameters.w, 24 )
};
struct ShadingData4 { float4 data0, data1; uint4 data2; /* for fast 128-bit access */ };

// random numbers

LH2_DEVFUNC __inline__ uint WangHash( uint s ) { s = (s ^ 61) ^ (s >> 16), s *= 9, s = s ^ (s >> 4), s *= 0x27d4eb2d, s = s ^ (s >> 15); return s; }
LH2_DEVFUNC __inline__ uint RandomInt( uint& s ) { s ^= s << 13, s ^= s >> 17, s ^= s << 5; return s; }
LH2_DEVFUNC __inline__ float RandomFloat( uint& s ) { return RandomInt( s ) * 2.3283064365387e-10f; }

// math helpers

LH2_DEVFUNC float3 min3( const float3& a, const float3& b ) { return make_float3( min( a.x, b.x ), min( a.y, b.y ), min( a.z, b.z ) ); }
LH2_DEVFUNC float3 min3( const float3& a, const float b ) { return make_float3( min( a.x, b ), min( a.y, b ), min( a.z, b ) ); }
LH2_DEVFUNC float3 min3( const float a, const float3& b ) { return make_float3( min( a, b.x ), min( a, b.y ), min( a, b.z ) ); }
LH2_DEVFUNC float3 max3( const float3& a, const float3& b ) { return make_float3( max( a.x, b.x ), max( a.y, b.y ), max( a.z, b.z ) ); }
LH2_DEVFUNC float3 max3( const float3& a, const float b ) { return make_float3( max( a.x, b ), max( a.y, b ), max( a.z, b ) ); }
LH2_DEVFUNC float3 max3( const float a, const float3& b ) { return make_float3( max( a, b.x ), max( a, b.y ), max( a, b.z ) ); }
LH2_DEVFUNC float4 max4( const float4& a, const float4& b ) { return make_float4( max( a.x, b.x ), max( a.y, b.y ), max( a.z, b.z ), max( a.w, b.w ) ); }
LH2_DEVFUNC float sqrLength( const float3& a ) { return dot( a, a ); }
LH2_DEVFUNC float sqr( const float x ) { return x * x; }
LH2_DEVFUNC float oneoverpow2( int p ) { return __uint_as_float( (127 - p) << 23 ); }
LH2_DEVFUNC float fastexpf( float x )
{	// https://codingforspeed.com/using-faster-exponential-approximation
	x = 1.0f + x / 256.0f;
	x *= x; x *= x; x *= x; x *= x;
	x *= x; x *= x; x *= x; x *= x;
	return x;
}

// from: https://aras-p.info/texts/CompactNormalStorage.html
LH2_DEVFUNC uint PackNormal( float3 N )
{
#if 1
	// more efficient
	const float f = 65535.0f / sqrtf( 8.0f * N.z + 8.0f );
	return (uint)(N.x * f + 32767.0f) + ((uint)(N.y * f + 32767.0f) << 16);
#else
	float2 enc = normalize( make_float2( N ) ) * (sqrtf( -N.z * 0.5f + 0.5f ));
	enc = enc * 0.5f + 0.5f;
	return (uint)(enc.x * 65535.0f) + ((uint)(enc.y * 65535.0f) << 16);
#endif
}
LH2_DEVFUNC float3 UnpackNormal( uint p )
{
	float4 nn = make_float4( (float)(p & 65535) * (2.0f / 65535.0f), (float)(p >> 16) * (2.0f / 65535.0f), 0, 0 );
	nn += make_float4( -1, -1, 1, -1 );
	float l = dot( make_float3( nn.x, nn.y, nn.z ), make_float3( -nn.x, -nn.y, -nn.w ) );
	nn.z = l, l = sqrtf( l ), nn.x *= l, nn.y *= l;
	return make_float3( nn ) * 2.0f + make_float3( 0, 0, -1 );
}
// alternative method
LH2_DEVFUNC uint PackNormal2( float3 N )
{
	// simple, and good enough discrimination of normals for filtering.
	const uint x = clamp( (uint)((N.x + 1) * 511), 0u, 1023u );
	const uint y = clamp( (uint)((N.y + 1) * 511), 0u, 1023u );
	const uint z = clamp( (uint)((N.z + 1) * 511), 0u, 1023u );
	return (x << 2u) + (y << 12u) + (z << 22u);
}
LH2_DEVFUNC float3 UnpackNormal2( uint pi )
{
	const uint x = (pi >> 2u) & 1023u;
	const uint y = (pi >> 12u) & 1023u;
	const uint z = pi >> 22u;
	return make_float3( x * (1.0f / 511.0f) - 1, y * (1.0f / 511.0f) - 1, z * (1.0f / 511.0f) - 1 );
}

// color conversions

LH2_DEVFUNC float3 RGBToYCoCg( const float3 RGB )
{
	const float3 rgb = min3( make_float3( 4 ), RGB ); // clamp helps AA for strong HDR
	const float Y = dot( rgb, make_float3( 1, 2, 1 ) ) * 0.25f;
	const float Co = dot( rgb, make_float3( 2, 0, -2 ) ) * 0.25f + (0.5f * 256.0f / 255.0f);
	const float Cg = dot( rgb, make_float3( -1, 2, -1 ) ) * 0.25f + (0.5f * 256.0f / 255.0f);
	return make_float3( Y, Co, Cg );
}

LH2_DEVFUNC float3 YCoCgToRGB( const float3 YCoCg )
{
	const float Y = YCoCg.x;
	const float Co = YCoCg.y - (0.5f * 256.0f / 255.0f);
	const float Cg = YCoCg.z - (0.5f * 256.0f / 255.0f);
	return make_float3( Y + Co - Cg, Y + Cg, Y - Co - Cg );
}

LH2_DEVFUNC float Luminance( float3 rgb )
{
	return 0.299f * min( rgb.x, 10.0f ) + 0.587f * min( rgb.y, 10.0f ) + 0.114f * min( rgb.z, 10.0f );
}

LH2_DEVFUNC uint HDRtoRGB32( const float3& c )
{
	const uint r = (uint)(1023.0f * min( 1.0f, c.x ));
	const uint g = (uint)(2047.0f * min( 1.0f, c.y ));
	const uint b = (uint)(2047.0f * min( 1.0f, c.z ));
	return (r << 22) + (g << 11) + b;
}
LH2_DEVFUNC float3 RGB32toHDR( const uint c )
{
	return make_float3(
		(float)(c >> 22)  * (1.0f / 1023.0f),
		(float)((c >> 11) & 2047) * (1.0f / 2047.0f),
		(float)(c & 2047) * (1.0f / 2047.0f)
	);
}
LH2_DEVFUNC float3 RGB32toHDRmin1( const uint c )
{
	return make_float3(
		(float)max( 1u, c >> 22 ) * (1.0f / 1023.0f),
		(float)max( 1u, (c >> 11) & 2047 ) * (1.0f / 2047.0f),
		(float)max( 1u, c & 2047 ) * (1.0f / 2047.0f) );
}

LH2_DEVFUNC float4 SampleSkydome( float3 D, const int pathLength )
{
	// formulas by Paul Debevec, http://www.pauldebevec.com/Probes
	uint u = (uint)(skywidth * 0.5f * (1.0f + atan2( D.x, -D.z ) * INVPI));
	uint v = (uint)(skyheight * acos( D.y ) * INVPI);
	uint idx = u + v * skywidth;
	return idx < skywidth * skyheight ? make_float4( skyPixels[idx], 1.0f ) : make_float4( 0 );
}

LH2_DEVFUNC float SurvivalProbability( const float3& diffuse )
{
	return min( 1.0f, max( max( diffuse.x, diffuse.y ), diffuse.z ) );
}

LH2_DEVFUNC float FresnelDielectricExact( const float3& wo, const float3& N, float eta )
{
	if (eta <= 1.0f) return 0.0f;
	const float cosThetaI = max( 0.0f, dot( wo, N ) );
	float scale = 1 / eta, cosThetaTSqr = 1 - (1 - cosThetaI * cosThetaI) * (scale * scale);
	if (cosThetaTSqr <= 0.0f) return 1.0f;
	float cosThetaT = sqrtf( cosThetaTSqr );
	float Rs = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
	float Rp = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
	return 0.5f * (Rs * Rs + Rp * Rp);
}

LH2_DEVFUNC float3 Tangent2World( const float3& V, const float3& N )
{
	// "Building an Orthonormal Basis, Revisited"
	float sign = copysignf( 1.0f, N.z );
	const float a = -1.0f / (sign + N.z);
	const float b = N.x * N.y * a;
	const float3 B = make_float3( 1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x );
	const float3 T = make_float3( b, sign + N.y * N.y * a, -N.y );
	return V.x * T + V.y * B + V.z * N;
}

LH2_DEVFUNC float3 World2Tangent( const float3& __restrict__ V, const float3& __restrict__ N )
{
	float sign = copysignf( 1.0f, N.z );
	const float a = -1.0f / (sign + N.z);
	const float b = N.x * N.y * a;
	const float3 B = make_float3( 1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x );
	const float3 T = make_float3( b, sign + N.y * N.y * a, -N.y );
	return make_float3( dot( V, T ), dot( V, B ), dot( V, N ) );
}

LH2_DEVFUNC float3 DiffuseReflectionUniform( const float r0, const float r1 )
{
	const float term1 = TWOPI * r0, term2 = sqrtf( 1 - r1 * r1 );
	float s, c;
	__sincosf( term1, &s, &c );
	return make_float3( c * term2, s * term2, r1 );
}

LH2_DEVFUNC float3 DiffuseReflectionCosWeighted( const float r0, const float r1 )
{
	const float term1 = TWOPI * r0, term2 = sqrtf( 1 - r1 );
	float s, c;
	__sincosf( term1, &s, &c );
	return make_float3( c * term2, s * term2, sqrtf( r1 ) );
}

// origin offset

LH2_DEVFUNC float3 SafeOrigin( const float3& O, const float3& R, const float3& N, const float geoEpsilon )
{
	// offset outgoing ray direction along R and / or N: along N when strongly parallel to the origin surface; mostly along R otherwise
	const float parallel = 1 - fabs( dot( N, R ) );
	const float v = parallel * parallel;
#if 0
	// we can go slightly into the surface when iN != N; negate the offset along N in that case
	const float side = dot( N, R ) < 0 ? -1 : 1;
#else
	// negating offset along N only makes sense once we backface cull
	const float side = 1.0f;
#endif
	return O + R * geoEpsilon * (1 - v) + N * side * geoEpsilon * v;
}

// consistent normal interpolation

LH2_DEVFUNC float3 ConsistentNormal( const float3& D, const float3& iN, const float alpha )
{
	// part of the implementation of "Consistent Normal Interpolation", Reshetov et al., 2010
	// calculates a safe normal given an incoming direction, phong normal and alpha
#ifndef CONSISTENTNORMALS
	return iN;
#else
#if 0
	// Eq. 1, exact
	const float q = (1 - sinf( alpha )) / (1 + sinf( alpha ));
#else
	// Eq. 1 approximation, as in Figure 6 (not the wrong one in Table 8)
	const float t = PI - 2 * alpha, q = (t * t) / (PI * (PI + (2 * PI - 4) * alpha));
#endif
	const float b = dot( D, iN ), g = 1 + q * (b - 1), rho = sqrtf( q * (1 + g) / (1 + b) );
	const float3 Rc = (g + rho * b) * iN - (rho * D);
	return normalize( D + Rc );
#endif
}

LH2_DEVFUNC float4 CombineToFloat4( const float3& A, const float3& B )
{
	// convert two float4's to a single uint4, where each int stores two components of the input vectors.
	// assumptions:
	// - the input is possitive
	// - the input can be safely clamped to 31.999
	// with this in mind, the data is stored as 5:11 unsigned fixed point, which should be plenty.
	const uint Ar = (uint)(min( A.x, 31.999f ) * 2048.0f), Ag = (uint)(min( A.y, 31.999f ) * 2048.0f), Ab = (uint)(min( A.z, 31.999f ) * 2048.0f);
	const uint Br = (uint)(min( B.x, 31.999f ) * 2048.0f), Bg = (uint)(min( B.y, 31.999f ) * 2048.0f), Bb = (uint)(min( B.z, 31.999f ) * 2048.0f);
	return make_float4( __uint_as_float( (Ar << 16) + Ag ), __uint_as_float( Ab ), __uint_as_float( (Br << 16) + Bg ), __uint_as_float( Bb ) );
}

LH2_DEVFUNC float3 GetDirectFromFloat4( const float4& X )
{
	const uint v0 = __float_as_uint( X.x ), v1 = __float_as_uint( X.y );
	return make_float3( (float)(v0 >> 16) * (1.0f / 2048.0f), (float)(v0 & 65535) * (1.0f / 2048.0f), (float)v1 * (1.0f / 2048.0f) );
}

LH2_DEVFUNC float3 GetIndirectFromFloat4( const float4& X )
{
	const uint v2 = __float_as_uint( X.z ), v3 = __float_as_uint( X.w );
	return make_float3( (float)(v2 >> 16) * (1.0f / 2048.0f), (float)(v2 & 65535) * (1.0f / 2048.0f), (float)v3 * (1.0f / 2048.0f) );
}

LH2_DEVFUNC float blueNoiseSampler( const uint* blueNoise, int x, int y, int sampleIndex, int sampleDimension )
{
	// wrap arguments
	x &= 127, y &= 127, sampleIndex &= 255, sampleDimension &= 255;

	// xor index based on optimized ranking
	int rankedSampleIndex = sampleIndex ^ blueNoise[sampleDimension + (x + y * 128) * 8 + 65536 * 3];

	// fetch value in sequence
	int value = blueNoise[sampleDimension + rankedSampleIndex * 256];

	// if the dimension is optimized, xor sequence value based on optimized scrambling
	value ^= blueNoise[(sampleDimension & 7) + (x + y * 128) * 8 + 65536];

	// convert to float and return
	return (0.5f + value) * (1.0f / 256.0f);
}

LH2_DEVFUNC void getPathInfo(const uint& path_s_t_type_pass, uint& pass, uint& s, uint& t, uint& type)
{
    pass = (path_s_t_type_pass & 524287);
    type = (path_s_t_type_pass >> 19);
    t = (type >> 3);
    s = (t >> 5);
    type = (type & 7);
    t = (t & 31);
}

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

// EOF
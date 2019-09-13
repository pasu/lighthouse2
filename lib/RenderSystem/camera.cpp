/* camera.cpp - Copyright 2019 Utrecht University

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

#include "rendersystem.h"

//  +-----------------------------------------------------------------------------+
//  |  Camera::Camera                                                             |
//  |  Constructor.                                                         LH2'19|
//  +-----------------------------------------------------------------------------+
Camera::Camera( const char* xmlFile )
{
	Deserialize( xmlFile );
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::~Camera                                                            |
//  |  Destructor.                                                          LH2'19|
//  +-----------------------------------------------------------------------------+
Camera::~Camera()
{
	Serialize( xmlFile.c_str() );
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::CalculateMatrix                                                    |
//  |  Helper function; constructs camera matrix.                           LH2'19|
//  +-----------------------------------------------------------------------------+
void Camera::CalculateMatrix( float3& x, float3& y, float3& z )
{
	y = make_float3( 0, 1, 0 );
	z = direction; // assumed to be normalized at all times
	x = normalize( cross( z, y ) );
	y = cross( x, z );
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::LookAt                                                             |
//  |  Position and aim the camera.                                         LH2'19|
//  +-----------------------------------------------------------------------------+
void Camera::LookAt( const float3 O, const float3 T )
{
	position = O;
	direction = normalize( T - O );
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::TranslateRelative                                                  |
//  |  Move the camera with respect to the current orientation.             LH2'19|
//  +-----------------------------------------------------------------------------+
void Camera::TranslateRelative( float3 T )
{
	float3 right, up, forward;
	CalculateMatrix( right, up, forward );
	float3 delta = T.x * right + T.y * up + T.z * forward;
	position += delta;
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::TranslateTarget                                                    |
//  |  Move the camera target with respect to the current orientation.      LH2'19|
//  +-----------------------------------------------------------------------------+
void Camera::TranslateTarget( float3 T )
{
	float3 right, up, forward;
	CalculateMatrix( right, up, forward );
	direction = normalize( direction + T.x * right + T.y * up + T.z * forward );
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::GetView                                                            |
//  |  Create a ViewPyramid for rendering in the RenderCore layer.          LH2'19|
//  +-----------------------------------------------------------------------------+
ViewPyramid Camera::GetView()
{
	ViewPyramid view;
	float3 right, up, forward;
	CalculateMatrix( right, up, forward );
	view.pos = position;
	view.spreadAngle = (FOV * PI / 180) / (float)pixelCount.y;
	const float screenSize = tanf( FOV / 2 / (180 / PI) );
	const float3 C = view.pos + focalDistance * forward;
	view.p1 = C - screenSize * right * focalDistance * aspectRatio + screenSize * focalDistance * up;
	view.p2 = C + screenSize * right * focalDistance * aspectRatio + screenSize * focalDistance * up;
	view.p3 = C - screenSize * right * focalDistance * aspectRatio - screenSize * focalDistance * up;
	view.aperture = aperture;

//     float3 w = view.p1 - view.p2;
//     float w1 = w.x*w.x + w.y*w.y + w.z*w.z;
//     float3 h = view.p1 - view.p3;
//     float h1 = h.x*h.x + h.y*h.y + h.z*h.z;
//     float3 p = view.p2 - view.p3;
//     float p1 = p.x*p.x + p.y*p.y + p.z*p.z;
	return view;
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::Serialize                                                          |
//  |  Save the camera data to the specified xml file.                      LH2'19|
//  +-----------------------------------------------------------------------------+
void Camera::Serialize( const char* xmlFileName )
{
	XMLDocument doc;
	XMLNode* root = doc.NewElement( "camera" );
	doc.InsertFirstChild( root );
	XMLElement* campos = doc.NewElement( "position" );
	campos->SetAttribute( "x", position.x );
	campos->SetAttribute( "y", position.y );
	campos->SetAttribute( "z", position.z );
	root->InsertEndChild( campos );
	XMLElement* camdir = doc.NewElement( "direction" );
	camdir->SetAttribute( "x", direction.x );
	camdir->SetAttribute( "y", direction.y );
	camdir->SetAttribute( "z", direction.z );
	root->InsertEndChild( camdir );
	((XMLElement*)root->InsertEndChild( doc.NewElement( "FOV" ) ))->SetText( FOV );
	((XMLElement*)root->InsertEndChild( doc.NewElement( "brightness" ) ))->SetText( brightness );
	((XMLElement*)root->InsertEndChild( doc.NewElement( "contrast" ) ))->SetText( contrast );
	((XMLElement*)root->InsertEndChild( doc.NewElement( "aperture" ) ))->SetText( aperture );
	((XMLElement*)root->InsertEndChild( doc.NewElement( "focalDistance" ) ))->SetText( focalDistance );
	((XMLElement*)root->InsertEndChild( doc.NewElement( "clampValue" ) ))->SetText( clampValue );
	doc.SaveFile( xmlFileName ? xmlFileName : xmlFile.c_str() );
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::Deserialize                                                        |
//  |  Load the camera data from the specified xml file.                    LH2'19|
//  +-----------------------------------------------------------------------------+
void Camera::Deserialize( const char* xmlFileName )
{
	xmlFile = xmlFileName;
	XMLDocument doc;
	XMLError result = doc.LoadFile( "camera.xml" );
	if (result != XML_SUCCESS) return;
	XMLNode* root = doc.FirstChild();
	if (root == nullptr) return;
	XMLElement* element = root->FirstChildElement( "position" );
	if (!element) return;
	element->QueryFloatAttribute( "x", &position.x );
	element->QueryFloatAttribute( "y", &position.y );
	element->QueryFloatAttribute( "z", &position.z );
	element = root->FirstChildElement( "direction" );
	if (!element) return;
	element->QueryFloatAttribute( "x", &direction.x );
	element->QueryFloatAttribute( "y", &direction.y );
	element->QueryFloatAttribute( "z", &direction.z );
	root->FirstChildElement( "FOV" )->QueryFloatText( &FOV );
	root->FirstChildElement( "brightness" )->QueryFloatText( &brightness );
	root->FirstChildElement( "contrast" )->QueryFloatText( &contrast );
	root->FirstChildElement( "aperture" )->QueryFloatText( &aperture );
	root->FirstChildElement( "focalDistance" )->QueryFloatText( &focalDistance );
	root->FirstChildElement( "clampValue" )->QueryFloatText( &clampValue );
}

// EOF
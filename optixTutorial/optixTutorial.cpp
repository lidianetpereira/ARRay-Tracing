#pragma clang diagnostic push
#pragma ide diagnostic ignored "bugprone-narrowing-conversions"
/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined( _WIN32 )
#  include <GL/wglew.h>
#  include <GL/freeglut.h>
#  else
#  include <GL/glut.h>
#  endif
#endif

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include <sutil.h>
#include "common.h"
#include "random.h"
#include <Arcball.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <math.h>
#include <stdint.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef DEBUG
#  ifdef _WIN32
#    define MAXPATHLEN MAX_PATH
#    include <direct.h>               // _getcwd()
#    define getcwd _getcwd
#  else
#    include <unistd.h>
#    include <sys/param.h>
#  endif
#    include <unistd.h>
#    include <sys/param.h>
#endif
#include <ARX/ARController.h>
#include <ARX/ARUtil/time.h>
#include <GL/gl.h>
#include "draw.h"
#include <ARX/ARG/mtx.h>
#include <optixu/optixu_quaternion.h>
#include <OptiXMesh.h>

#if ARX_TARGET_PLATFORM_WINDOWS
const char *vconf = "-module=WinMF -format=BGRA";
#else
const char *vconf = NULL;
#endif
const char *cpara = NULL;

#define ar 1

using namespace optix;

const char* const SAMPLE_NAME = "optixTutorial";

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context      context;
uint32_t     width  = 960u;
uint32_t     height = 720u;

std::string  texture_path;
int          tutorial_number = 3;

bool   m_interop;
GLuint m_pbo;
GLuint m_tex;
bool  use_tri_api = true;
bool  ignore_mats = false;

Buffer m_buffer;

// Viewport size
int m_width = 960;
int m_height = 720;

// Camera state
float3       camera_up;
float3       camera_lookat;
float3       camera_eye;
float3       camera_eyeOld;
Matrix4x4    camera_rotate;
sutil::Arcball arcball;
bool  camera_dirty = true;

// Mouse state
int2       mouse_prev_pos;
int        mouse_button;

//--AR
static int contextWidth = 0;
static int contextHeight = 0;
static bool contextWasUpdated = true;
static int32_t viewport[4];
static float projection[16];

static ARController* arController = NULL;
static ARG_API drawAPI = ARG_API_GL3;

static long gFrameNo = 0;

struct marker {
    const char *name;
    float height;
};
static const struct marker markers[] = {
        {"hiro.patt", 57.0},
        {"kanji.patt", 57.0}
};
static const int markerCount = (sizeof(markers)/sizeof(markers[0]));

// Add trackables.
int markerIDs[markerCount];
int markerModelIDs[markerCount];

char str[512];
float invOut[16];

Group m_top_object;
optix::Aabb  aabb;

Program pgram_intersection = 0;
Program pgram_bounding_box = 0;
Program diffuse_ch = 0;
Program diffuse_ah = 0;

Transform cornellPose;
Transform scale;
float transformMat[16];

int scene = 6;
//------------------------------------------------------------------------------
//
// Forward decls
//
//------------------------------------------------------------------------------

Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
void createContext();
void createGeometry();
void setupCamera();
void setupLights();
void updateCamera();


void glutInitialize( int* argc, char** argv );
void glutRun();
void glutDisplay();
void glutKeyboardPress( unsigned char k, int x, int y );
void glutMousePress( int button, int state, int x, int y );
void glutMouseMotion( int x, int y);
void glutResize( int w, int h );

//--AR
static void quit(int rc);
static void reshape(int w, int h);
static void init();
void showString(std::string str);
bool gluInvertMatrix(float m[16]);
static void displayOnce(void);
bool distanceBigger(float3 P, float3 Q);

//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

bool distanceBigger(float3 P, float3 Q){
    float threshold = 0.5f;

    float distance = sqrt(pow(P.x - Q.x, 2.0)+ pow(P.y - Q.y, 2.0) + pow(P.z - Q.z, 2.0));
    //printf("d = %f\n", distance);
    return distance > threshold;
}


Buffer getOutputBuffer()
{
    return context[ "output_buffer" ]->getBuffer();
}


void destroyContext()
{
    if( context )
    {
        context->destroy();
        context = 0;
    }
}


void registerExitHandler()
{
    // register shutdown handler
#ifdef _WIN32
    glutCloseFunc( destroyContext );  // this function is freeglut-only
#else
    atexit( destroyContext );
#endif
}


void createContext()
{
    // Set up context
    context = Context::create();
    context->setRayTypeCount( 2 );
    context->setEntryPointCount( 1 );
    context->setStackSize( 2800 );
    context->setMaxTraceDepth( 12 );

    // Note: high max depth for reflection and refraction through glass
    context["max_depth"]->setInt( 10 );
    context["frame"]->setUint( 0u );
    context["scene_epsilon"]->setFloat( 1.e-4f );
    context["ambient_light_color"]->setFloat( 0.8f, 0.8f, 0.8f );

    // OptiX buffer initialization:
    m_buffer = (m_interop) ? context->createBufferFromGLBO(RT_BUFFER_OUTPUT, m_pbo)
                           : context->createBuffer(RT_BUFFER_OUTPUT);
    m_buffer->setFormat(RT_FORMAT_UNSIGNED_BYTE4); // BGRA8
    m_buffer->setSize(width, height);
    context["output_buffer"]->set(m_buffer);

    // Accumulation buffer.  This scene has a lot of high frequency detail and
    // benefits from accumulation of samples.
    Buffer accum_buffer = context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
                                                 RT_FORMAT_FLOAT4, width, height );
    context["accum_buffer"]->set( accum_buffer );

    // Ray generation program
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "accum_camera.cu" );
    Program ray_gen_program = context->createProgramFromPTXString( ptx, "pinhole_camera" );
    context->setRayGenerationProgram( 0, ray_gen_program );

    // Exception program
    Program exception_program = context->createProgramFromPTXString( ptx, "exception" );
    context->setExceptionProgram( 0, exception_program );
    context["bad_color"]->setFloat( 1.0f, 0.0f, 1.0f );

    // Miss program
    context->setMissProgram( 0, context->createProgramFromPTXString( sutil::getPtxString( SAMPLE_NAME, "constantbg.cu" ), "miss" ) );
    context["bg_color"]->setFloat( 1.0f, 1.0f, 1.0f );
}


void setMaterial(GeometryInstance& gi, Material material)
{
    gi->addMaterial(material);
}


Material createMaterial(const float3& color){

    Material diffuse = context->createMaterial();
    diffuse->setClosestHitProgram( 0, diffuse_ch );
    diffuse->setAnyHitProgram( 1, diffuse_ah );
    diffuse["Kd1"]->setFloat( color);
    diffuse["Ka1"]->setFloat( color);
    diffuse["Ks1"]->setFloat( 0.0f, 0.0f, 0.0f);
    diffuse["Kd2"]->setFloat( color);
    diffuse["Ka2"]->setFloat( color);
    diffuse["Ks2"]->setFloat( 0.0f, 0.0f, 0.0f);
    diffuse["inv_checker_size"]->setFloat( 32.0f, 16.0f, 1.0f );
    diffuse["phong_exp1"]->setFloat( 0.0f );
    diffuse["phong_exp2"]->setFloat( 0.0f );
    diffuse["Kr1"]->setFloat( 0.0f, 0.0f, 0.0f);
    diffuse["Kr2"]->setFloat( 0.0f, 0.0f, 0.0f);

    return diffuse;
}


GeometryInstance createParallelogram( const float3& anchor, const float3& offset1, const float3& offset2)
{
    Geometry parallelogram = context->createGeometry();
    parallelogram->setPrimitiveCount( 1u );
    parallelogram->setIntersectionProgram( pgram_intersection );
    parallelogram->setBoundingBoxProgram( pgram_bounding_box );

    float3 normal = normalize( cross( offset1, offset2 ) );
    float d = dot( normal, anchor );
    float4 plane = make_float4( normal, d );

    float3 v1 = offset1 / dot( offset1, offset1 );
    float3 v2 = offset2 / dot( offset2, offset2 );

    parallelogram["plane"]->setFloat( plane );
    parallelogram["anchor"]->setFloat( anchor );
    parallelogram["v1"]->setFloat( v1 );
    parallelogram["v2"]->setFloat( v2 );

    GeometryInstance gi = context->createGeometryInstance();
    gi->setGeometry(parallelogram);
    return gi;
}


GeometryInstance loadMesh(const std::string& filename)
{
    OptiXMesh mesh;
    mesh.context = context;
    mesh.use_tri_api = use_tri_api;
    mesh.ignore_mats = false;
    loadMesh(filename, mesh);
    aabb.set(mesh.bbox_min, mesh.bbox_max);

    return mesh.geom_instance;
}


void createGeometry()
{
    m_top_object = context->createGroup();
    m_top_object->setAcceleration( context->createAcceleration("Trbvh"));

    // Create glass sphere geometry
    Geometry glass_sphere = context->createGeometry();
    glass_sphere->setPrimitiveCount( 1u );

    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "sphere_shell.cu" );
    glass_sphere->setBoundingBoxProgram( context->createProgramFromPTXString( ptx, "bounds" ) );
    glass_sphere->setIntersectionProgram( context->createProgramFromPTXString( ptx, "intersect" ) );
    glass_sphere["center"]->setFloat( 120.0f, 100.0f, 100.0f );
    glass_sphere["radius1"]->setFloat( 99.6f );
    glass_sphere["radius2"]->setFloat( 100.0f );

    // Metal sphere geometry
    Geometry metal_sphere = context->createGeometry();
    metal_sphere->setPrimitiveCount( 1u );
    ptx = sutil::getPtxString( SAMPLE_NAME, "sphere.cu" );
    metal_sphere->setBoundingBoxProgram( context->createProgramFromPTXString( ptx, "bounds" ) );
    metal_sphere->setIntersectionProgram( context->createProgramFromPTXString( ptx, "robust_intersect" ) );
    metal_sphere["sphere"]->setFloat( -120.0f, 100.0f, -100.0f, 100.0f );

    // Set up parallelogram programs
    ptx = sutil::getPtxString( SAMPLE_NAME, "parallelogram.cu" );
    pgram_bounding_box = context->createProgramFromPTXString( ptx, "bounds" );
    pgram_intersection = context->createProgramFromPTXString( ptx, "intersect" );

    // Glass material
    ptx = sutil::getPtxString( SAMPLE_NAME, "glass.cu" );
    Program glass_ch = context->createProgramFromPTXString( ptx, "closest_hit_radiance" );
    Program glass_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );
    Material glass_matl = context->createMaterial();
    glass_matl->setClosestHitProgram( 0, glass_ch );
    glass_matl->setAnyHitProgram( 1, glass_ah );
    glass_matl["importance_cutoff"]->setFloat( 1e-2f );
    glass_matl["cutoff_color"]->setFloat( 0.034f, 0.055f, 0.085f );
    glass_matl["fresnel_exponent"]->setFloat( 3.0f );
    glass_matl["fresnel_minimum"]->setFloat( 0.1f );
    glass_matl["fresnel_maximum"]->setFloat( 1.0f );
    glass_matl["refraction_index"]->setFloat( 1.4f );
    glass_matl["refraction_color"]->setFloat( 1.0f, 1.0f, 1.0f );
    glass_matl["reflection_color"]->setFloat( 1.0f, 1.0f, 1.0f );
    glass_matl["refraction_maxdepth"]->setInt( 10 );
    glass_matl["reflection_maxdepth"]->setInt( 5 );
    const float3 extinction = make_float3(.83f, .83f, .83f);
    glass_matl["extinction_constant"]->setFloat( log(extinction.x), log(extinction.y), log(extinction.z) );
    glass_matl["shadow_attenuation"]->setFloat( 0.6f, 0.6f, 0.6f );

    Material colored_glass = context->createMaterial();
    colored_glass->setClosestHitProgram( 0, glass_ch );
    colored_glass["fresnel_exponent"]->setFloat( 4.0f );
    colored_glass["fresnel_minimum"]->setFloat( 0.1f );
    colored_glass["fresnel_maximum"]->setFloat( 1.0f );
    colored_glass["refraction_index"]->setFloat( 1.4f );
    colored_glass["refraction_color"]->setFloat( 0.99f, 0.99f, 0.99f );
    colored_glass["reflection_color"]->setFloat( 0.99f, 0.99f, 0.99f );
    colored_glass["extinction_constant"]->setFloat( log(extinction.x), log(extinction.y), log(extinction.z) );

    // Metal material
    ptx = sutil::getPtxString( SAMPLE_NAME, "phong.cu" );
    Program phong_ch = context->createProgramFromPTXString( ptx, "closest_hit_radiance" );
    Program phong_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );
    Material metal_matl = context->createMaterial();
    metal_matl->setClosestHitProgram( 0, phong_ch );
    metal_matl->setAnyHitProgram( 1, phong_ah );
    metal_matl["Ka"]->setFloat( 0.2f, 0.5f, 0.5f );
    metal_matl["Kd"]->setFloat( 0.2f, 0.7f, 0.8f );
    metal_matl["Ks"]->setFloat( 0.9f, 0.9f, 0.9f );
    metal_matl["phong_exp"]->setFloat( 64 );
    metal_matl["Kr"]->setFloat( 0.5f,  0.5f,  0.5f);

    ptx = sutil::getPtxString( SAMPLE_NAME, "checker.cu" );
    diffuse_ch = context->createProgramFromPTXString( ptx, "closest_hit_radiance" );
    diffuse_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );

    const float3 white = make_float3( 0.8f, 0.8f, 0.8f );
    const float3 green = make_float3( 0.05f, 0.8f, 0.05f );
    const float3 red   = make_float3( 0.8f, 0.05f, 0.05f );
    const float3 light_em = make_float3( 0.9f, 0.9f, 0.9f );

    // Create GIs for each piece of geometry
    std::vector<GeometryInstance> gis;

    // Floor
    gis.push_back( createParallelogram( make_float3( -278.0f, 0.0f, -279.6f ),
                                        make_float3( 0.0f, 0.0f, 559.2f ),
                                        make_float3( 556.0f, 0.0f, 0.0f ) ) );
    setMaterial(gis.back(), createMaterial(white));

    // Ceiling
    gis.push_back( createParallelogram( make_float3( -278.0f, 548.8f, -279.6f ),
                                        make_float3( 556.0f, 0.0f, 0.0f ),
                                        make_float3( 0.0f, 0.0f, 559.2f ) ) );
    setMaterial(gis.back(), createMaterial(white));

    // Back wall
    gis.push_back( createParallelogram( make_float3( -278.0f, 0.0f, -279.6f),
                                        make_float3( 0.0f, 548.8f, 0.0f),
                                        make_float3( 556.0f, 0.0f, 0.0f) ) );
    setMaterial(gis.back(), createMaterial(white));

    // Right wall
    gis.push_back( createParallelogram( make_float3( -278.0f, 0.0f, -279.6f ),
                                        make_float3( 0.0f, 548.8f, 0.0f ),
                                        make_float3( 0.0f, 0.0f, 559.2f ) ) );
    setMaterial(gis.back(), createMaterial(green));

    // Left wall
    gis.push_back( createParallelogram( make_float3( 278.0f, 0.0f, -279.6f ),
                                        make_float3( 0.0f, 0.0f, 559.2f ),
                                        make_float3( 0.0f, 548.8f, 0.0f ) ) );
    setMaterial(gis.back(), createMaterial(red));

    // Light
    gis.push_back( createParallelogram( make_float3( 65.0f, 548.6f, -52.5f),
                                        make_float3( -130.0f, 0.0f, 0.0f),
                                        make_float3( 0.0f, 0.0f, 105.0f) ) );
    setMaterial(gis.back(), createMaterial(light_em));

    // Red metalic material
    ptx = sutil::getPtxString( SAMPLE_NAME, "phong.cu" );
    Material red_matl = context->createMaterial();
    red_matl->setClosestHitProgram( 0, phong_ch );
    red_matl->setAnyHitProgram( 1, phong_ah );
    red_matl["Ka"]->setFloat(0.5f, 0.2f, 0.1f);
    red_matl["Kd"]->setFloat(0.6f, 0.0f, 0.1f);
    red_matl["Ks"]->setFloat( 0.9f, 0.9f, 0.9f );
    red_matl["phong_exp"]->setFloat( 64 );
    red_matl["Kr"]->setFloat( 0.0f,  0.0f,  0.0f);

    std::vector<GeometryInstance> gisUnit;
    GeometryGroup geometrygroup, obj1, obj2, obj3, obj4;
    GeometryGroup geometry_groupMesh;
    Geometry glass_sphere2;

    scene = 0;

    switch(scene){
        case 0 : {
            gis.push_back(context->createGeometryInstance(glass_sphere, &glass_matl, &glass_matl + 1));
            gis.push_back(context->createGeometryInstance(metal_sphere, &metal_matl, &metal_matl + 1));

            geometrygroup = context->createGeometryGroup(gis.begin(), gis.end());
            geometrygroup->setAcceleration(context->createAcceleration("Trbvh"));

            mtxLoadIdentityf(transformMat);
            mtxScalef(transformMat, 0.2, 0.2, 0.2);
            mtxRotatef(transformMat, -90.0f, 1.0f, 0.0f, 0.0f);

            cornellPose = context->createTransform();
            cornellPose->setMatrix(false, transformMat, 0);

            cornellPose = context->createTransform();
            cornellPose->setMatrix(false, transformMat, 0);

            cornellPose->setChild(geometrygroup);
            m_top_object->addChild(cornellPose);

            context["top_object"]->set(m_top_object);
            context["top_shadower"]->set(m_top_object);
            break;
        } //two spheres and Cornell box
        default: {
            geometrygroup = context->createGeometryGroup(gis.begin(), gis.end());
            geometrygroup->setAcceleration(context->createAcceleration("Trbvh"));

            mtxLoadIdentityf(transformMat);
            mtxScalef(transformMat, 0.2, 0.2, 0.2);
            mtxRotatef(transformMat, -90.0f, 1.0f, 0.0f, 0.0f);

            cornellPose = context->createTransform();
            cornellPose->setMatrix(false, transformMat, 0);

            cornellPose->setChild(geometrygroup);
            m_top_object->addChild(cornellPose);

            context["top_object"]->set(m_top_object);
            context["top_shadower"]->set(m_top_object);
            break;
        } //only Cornell box
    }
}


void setupCamera()
{
    camera_eye    = make_float3( invOut[12], invOut[13], invOut[14] );
    camera_lookat = make_float3( camera_eye.x - invOut[8], camera_eye.y - invOut[9],  camera_eye.z - invOut[10] );
    camera_up     = make_float3( invOut[4], invOut[5], invOut[6]);

    camera_rotate  = Matrix4x4::identity();
    camera_dirty = true;
}


void setupLights()
{
    BasicLight lights[] = {
            { make_float3( 0.0f, 0.0f , 108.0f ), make_float3( 0.3f, 0.3f, 0.1f ), 1 }
    };

    Buffer light_buffer = context->createBuffer( RT_BUFFER_INPUT );
    light_buffer->setFormat( RT_FORMAT_USER );
    light_buffer->setElementSize( sizeof( BasicLight ) );
    light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
    memcpy(light_buffer->map(), lights, sizeof(lights));
    light_buffer->unmap();

    context[ "lights" ]->set( light_buffer );
}


void updateCamera()
{
    const float vfov  = 45.0f;
    const float aspect_ratio = static_cast<float>(width) /
                               static_cast<float>(height);

    float3 camera_u, camera_v, camera_w;
    sutil::calculateCameraVariables(
            camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
            camera_u, camera_v, camera_w, /*fov_is_vertical*/ true );

    const Matrix4x4 frame = Matrix4x4::fromBasis(
            normalize( camera_u ),
            normalize( camera_v ),
            normalize( -camera_w ),
            camera_lookat);
    const Matrix4x4 frame_inv = frame.inverse();
    // Apply camera rotation twice to match old SDK behavior
    const Matrix4x4 trans   = frame*camera_rotate*camera_rotate*frame_inv;

    camera_eye    = make_float3( trans*make_float4( camera_eye,    1.0f ) );
    camera_lookat = make_float3( trans*make_float4( camera_lookat, 1.0f ) );
    camera_up     = make_float3( trans*make_float4( camera_up,     0.0f ) );

    sutil::calculateCameraVariables(
            camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
            camera_u, camera_v, camera_w, true );

    camera_rotate = Matrix4x4::identity();

    context["eye"]->setFloat( camera_eye );
    context["U"  ]->setFloat( camera_u );
    context["V"  ]->setFloat( camera_v );
    context["W"  ]->setFloat( camera_w );

    camera_dirty = false;
}


void glutInitialize( int* argc, char** argv )
{
    glutInit( argc, argv );
    glutInitDisplayMode( GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize( width, height );
    glutInitWindowPosition( 100, 100 );
    glutCreateWindow( "ARRay - Tracing - OptiX" );
    glutHideWindow();
}


void glutRun()
{
    //Initialize GL state
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, width, height);

    if (m_interop)
    {
        glGenBuffers(1, &m_pbo);
        if(m_pbo != 0){ // Buffer size must be > 0 or OptiX can't create a buffer from it.
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
            glBufferData(GL_PIXEL_UNPACK_BUFFER, m_width * m_height * sizeof(unsigned char) * 4, nullptr, GL_STREAM_READ); // BRGA8
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        }else{
            ARLOGe("m_pbo tem tamanho zero");
        }
    }
    // glPixelStorei(GL_UNPACK_ALIGNMENT, 4); // default, works for BGRA8, RGBA16F, and RGBA32F.

    glGenTextures(1, &m_tex);
    if(m_tex != 0){
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_tex);

        // Change these to GL_LINEAR for super- or sub-sampling
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        // GL_CLAMP_TO_EDGE for linear filtering, not relevant for nearest.
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_BLEND);
    }else{
        ARLOGe("m_tex tem tamanho zero");
    }

    glutShowWindow();
    glutReshapeWindow(width, height);

    // register glut callbacks
    glutDisplayFunc( glutDisplay );
    glutIdleFunc( glutDisplay );
    glutReshapeFunc( glutResize );
    glutKeyboardFunc( glutKeyboardPress );
    glutMouseFunc( glutMousePress );
    glutMotionFunc( glutMouseMotion );

    registerExitHandler();

    glutMainLoop();
}


//------------------------------------------------------------------------------
//
//  GLUT callbacks
//
//------------------------------------------------------------------------------

void glutDisplay()
{
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    displayOnce();

    {
        static unsigned frame_count = 0;
        sutil::displayFps( frame_count++ );
    }

    float3 eyeTemp = make_float3(invOut[12], invOut[13], invOut[14]);
    if(distanceBigger(camera_eyeOld, eyeTemp)){
        camera_eyeOld = eyeTemp;
        camera_eye    = make_float3( invOut[12], invOut[13], invOut[14] );
        camera_lookat = make_float3( camera_eye.x - invOut[8], camera_eye.y - invOut[9],  camera_eye.z - invOut[10] );
        camera_up     = make_float3( invOut[4], invOut[5], invOut[6]);
        camera_dirty = true;
    }

    static unsigned int accumulation_frame = 0;
    if( camera_dirty ) {
        updateCamera();
        accumulation_frame = 0;
    }

    context["frame"]->setUint( accumulation_frame++ );
    context->launch( 0, width, height );

    // Update the OpenGL texture with the results:
    if (m_interop)
    {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_tex);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_buffer->getGLBOId());

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_width, m_height, 0, GL_BGRA, GL_UNSIGNED_BYTE, nullptr); // BGRA8
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }
    else
    {
        void const* data = m_buffer->map(0, RT_BUFFER_MAP_READ );
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, (GLsizei) width, (GLsizei) height, 0, GL_BGRA, GL_UNSIGNED_BYTE, data); // BGRA8
        m_buffer->unmap();
    }

    RTsize elmt_size = m_buffer->getElementSize();
    if      ( elmt_size % 8 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    else if ( elmt_size % 4 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    else if ( elmt_size % 2 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    else                          glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_tex);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
    drawTexConfig(m_tex);
    glDisable(GL_TEXTURE_2D);

    glutSwapBuffers();
}


void glutKeyboardPress( unsigned char k, int x, int y )
{

    switch( k )
    {
        case( 'q' ):
        case( 27 ): // ESC
        {
            drawCleanup();
            if (arController) {
                arController->drawVideoFinal(0);
                arController->shutdown();
                delete arController;
            }
            destroyContext();
            exit(0);
        }
        case( 's' ):
        {
            const std::string outputImage = std::string(SAMPLE_NAME) + ".ppm";
            std::cerr << "Saving current frame to '" << outputImage << "'\n";
            sutil::displayBufferPPM( outputImage.c_str(), getOutputBuffer() );
            break;
        }
    }
}


void glutMousePress( int button, int state, int x, int y )
{
    if( state == GLUT_DOWN )
    {
        mouse_button = button;
        mouse_prev_pos = make_int2( x, y );
    }
    else
    {
        // nothing
    }
}


void glutMouseMotion( int x, int y)
{
    if( mouse_button == GLUT_RIGHT_BUTTON )
    {
        const float dx = static_cast<float>( x - mouse_prev_pos.x ) /
                         static_cast<float>( width );
        const float dy = static_cast<float>( y - mouse_prev_pos.y ) /
                         static_cast<float>( height );
        const float dmax = fabsf( dx ) > fabs( dy ) ? dx : dy;
        const float scale = fminf( dmax, 0.9f );
        camera_eye = camera_eye + (camera_lookat - camera_eye)*scale;
        camera_dirty = true;
    }
    else if( mouse_button == GLUT_LEFT_BUTTON )
    {
        const float2 from = { static_cast<float>(mouse_prev_pos.x),
                              static_cast<float>(mouse_prev_pos.y) };
        const float2 to   = { static_cast<float>(x),
                              static_cast<float>(y) };

        const float2 a = { from.x / width, from.y / height };
        const float2 b = { to.x   / width, to.y   / height };

        camera_rotate = arcball.rotate( b, a );
        camera_dirty = true;
    }

    mouse_prev_pos = make_int2( x, y );
}


void glutResize( int w, int h )
{
    if ( w == (int)width && h == (int)height ) return;

    width  = w;
    height = h;
    sutil::ensureMinimumSize(width, height);

    sutil::resizeBuffer( getOutputBuffer(), width, height );
    sutil::resizeBuffer( context[ "accum_buffer" ]->getBuffer(), width, height );

    glViewport(0, 0, width, height);

    glutPostRedisplay();
}


//------------------------------------------------------------------------------
//
// AR
//
//------------------------------------------------------------------------------

static void init(){
#  if ARX_TARGET_PLATFORM_MACOS
    vconf = "-format=BGRA";
#  endif

    int w = 960, h = 720;
    reshape(w, h);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Initialise the ARController.
    arController = new ARController();
    if (!arController->initialiseBase()) {
        ARLOGe("Error initialising ARController.\n");
        quit(-1);
    }

#ifdef DEBUG
    arLogLevel = AR_LOG_LEVEL_DEBUG;
#endif

#ifdef DEBUG
    char buf[MAXPATHLEN];
    ARLOGd("CWD is '%s'.\n", getcwd(buf, sizeof(buf)));
#endif
    char *resourcesDir = "../";
    ARLOGd("Resources are in'%s'.\n", resourcesDir);
    for (int i = 0; i < markerCount; i++) {
        std::string markerConfig = "single;" + std::string(resourcesDir) + '/' + markers[i].name + ';' + std::to_string(markers[i].height);
        markerIDs[i] = arController->addTrackable(markerConfig);
        if (markerIDs[i] == -1) {
            ARLOGe("Error adding marker.\n");
            quit(-1);
        }
    }
    arController->getSquareTracker()->setPatternDetectionMode(AR_TEMPLATE_MATCHING_MONO);
    arController->getSquareTracker()->setThresholdMode(AR_LABELING_THRESH_MODE_AUTO_BRACKETING);

#ifdef DEBUG
    ARLOGd("vconf is '%s'.\n", vconf);
#endif
    arController->startRunning(vconf, cpara, NULL, 0);
}


static void displayOnce(void)
{

    // Main loop.
    bool done = false;
    while (!done) {
        bool gotFrame = arController->capture();
        if (!gotFrame) {
            arUtilSleep(1);
        } else {
            //ARLOGi("Got frame %ld.\n", gFrameNo);
            gFrameNo++;

            if (!arController->update()) {
                ARLOGe("Error in ARController::update().\n");
                quit(-1);
            }

            if (contextWasUpdated) {
                if (!arController->drawVideoInit(0)) {
                    ARLOGe("Error in ARController::drawVideoInit().\n");
                    quit(-1);
                }
                if (!arController->drawVideoSettings(0, contextWidth, contextHeight, false, false, false,
                                                     ARVideoView::HorizontalAlignment::H_ALIGN_CENTRE,
                                                     ARVideoView::VerticalAlignment::V_ALIGN_CENTRE,
                                                     ARVideoView::ScalingMode::SCALE_MODE_FIT, viewport)) {
                    ARLOGe("Error in ARController::drawVideoSettings().\n");
                    quit(-1);
                }
                drawSetup(drawAPI, false, false, false);
                drawSetViewport(viewport);
                ARdouble projectionARD[16];
                arController->projectionMatrix(0, 0.1f, 10000.0f, projectionARD);
                for (int i = 0; i < 16; i++) projection[i] = (float) projectionARD[i];
                drawSetCamera(projection, NULL);

                for (int i = 0; i < markerCount; i++) {
                    markerModelIDs[i] = drawLoadModel(NULL);
                }
                contextWasUpdated = false;
            }
            //Display the current video frame to the current OpenGL context.
            arController->drawVideo(0);

            // Look for trackables, and draw on each found one.
            for (int i = 0; i < markerCount; i++) {

                // Find the trackable for the given trackable ID.
                ARTrackable *marker = arController->findTrackable(markerIDs[i]);
                float view[16];
                if (marker->visible) {
                    //arUtilPrintMtx16(marker->transformationMatrix);
                    //ARLOGi("\n \n");
                    for (int i = 0; i < 16; i++){
                        view[i] = (float) marker->transformationMatrix[i];
                    }
                }
                if(gluInvertMatrix(view)){
                    sprintf(str, "Cam Pos: x: %3.1f  y: %3.1f  z: %3.1f w: %3.1f \n", invOut[12], invOut[13], invOut[14], invOut[15]);
                    //ARLOGd("Cam: x: %3.1f  y: %3.1f  z: %3.1f w: %3.1f \n", invOut[12], invOut[13], invOut[14], invOut[15]);
                }
            }
            done = true;
        }
    }
}


void showString(std::string str){
    int   i;

    for (i = 0; i < (int)str.length(); i++)
    {
        if(str[i] != '\n' )
            glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, str[i]);
        else
        {
            glRasterPos2i(0.0, 2.5);
        }
    }
}


bool gluInvertMatrix(float m[16])
{
    float inv[16], det;
    int i;

    inv[0] = m[5]  * m[10] * m[15] -
             m[5]  * m[11] * m[14] -
             m[9]  * m[6]  * m[15] +
             m[9]  * m[7]  * m[14] +
             m[13] * m[6]  * m[11] -
             m[13] * m[7]  * m[10];

    inv[4] = -m[4]  * m[10] * m[15] +
             m[4]  * m[11] * m[14] +
             m[8]  * m[6]  * m[15] -
             m[8]  * m[7]  * m[14] -
             m[12] * m[6]  * m[11] +
             m[12] * m[7]  * m[10];

    inv[8] = m[4]  * m[9] * m[15] -
             m[4]  * m[11] * m[13] -
             m[8]  * m[5] * m[15] +
             m[8]  * m[7] * m[13] +
             m[12] * m[5] * m[11] -
             m[12] * m[7] * m[9];

    inv[12] = -m[4]  * m[9] * m[14] +
              m[4]  * m[10] * m[13] +
              m[8]  * m[5] * m[14] -
              m[8]  * m[6] * m[13] -
              m[12] * m[5] * m[10] +
              m[12] * m[6] * m[9];

    inv[1] = -m[1]  * m[10] * m[15] +
             m[1]  * m[11] * m[14] +
             m[9]  * m[2] * m[15] -
             m[9]  * m[3] * m[14] -
             m[13] * m[2] * m[11] +
             m[13] * m[3] * m[10];

    inv[5] = m[0]  * m[10] * m[15] -
             m[0]  * m[11] * m[14] -
             m[8]  * m[2] * m[15] +
             m[8]  * m[3] * m[14] +
             m[12] * m[2] * m[11] -
             m[12] * m[3] * m[10];

    inv[9] = -m[0]  * m[9] * m[15] +
             m[0]  * m[11] * m[13] +
             m[8]  * m[1] * m[15] -
             m[8]  * m[3] * m[13] -
             m[12] * m[1] * m[11] +
             m[12] * m[3] * m[9];

    inv[13] = m[0]  * m[9] * m[14] -
              m[0]  * m[10] * m[13] -
              m[8]  * m[1] * m[14] +
              m[8]  * m[2] * m[13] +
              m[12] * m[1] * m[10] -
              m[12] * m[2] * m[9];

    inv[2] = m[1]  * m[6] * m[15] -
             m[1]  * m[7] * m[14] -
             m[5]  * m[2] * m[15] +
             m[5]  * m[3] * m[14] +
             m[13] * m[2] * m[7] -
             m[13] * m[3] * m[6];

    inv[6] = -m[0]  * m[6] * m[15] +
             m[0]  * m[7] * m[14] +
             m[4]  * m[2] * m[15] -
             m[4]  * m[3] * m[14] -
             m[12] * m[2] * m[7] +
             m[12] * m[3] * m[6];

    inv[10] = m[0]  * m[5] * m[15] -
              m[0]  * m[7] * m[13] -
              m[4]  * m[1] * m[15] +
              m[4]  * m[3] * m[13] +
              m[12] * m[1] * m[7] -
              m[12] * m[3] * m[5];

    inv[14] = -m[0]  * m[5] * m[14] +
              m[0]  * m[6] * m[13] +
              m[4]  * m[1] * m[14] -
              m[4]  * m[2] * m[13] -
              m[12] * m[1] * m[6] +
              m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] +
             m[1] * m[7] * m[10] +
             m[5] * m[2] * m[11] -
             m[5] * m[3] * m[10] -
             m[9] * m[2] * m[7] +
             m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] -
             m[0] * m[7] * m[10] -
             m[4] * m[2] * m[11] +
             m[4] * m[3] * m[10] +
             m[8] * m[2] * m[7] -
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] +
              m[0] * m[7] * m[9] +
              m[4] * m[1] * m[11] -
              m[4] * m[3] * m[9] -
              m[8] * m[1] * m[7] +
              m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] -
              m[0] * m[6] * m[9] -
              m[4] * m[1] * m[10] +
              m[4] * m[2] * m[9] +
              m[8] * m[1] * m[6] -
              m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (det == 0)
        return false;

    det = 1.0 / det;

    for (i = 0; i < 16; i++)
        invOut[i] = inv[i] * det;

    return true;
}


static void quit(int rc)
{
    drawCleanup();
    if (arController) {
        arController->drawVideoFinal(0);
        arController->shutdown();
        delete arController;
    }
    exit(rc);
}


static void reshape(int w, int h)
{
    contextWidth = w;
    contextHeight = h;
    ARLOGd("Resized to %dx%d.\n", w, h);
    contextWasUpdated = true;
}

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0 )
{
    std::cerr << "\nUsage: " << argv0 << " [options]\n";
    std::cerr <<
              "App Options:\n"
              "  -h | --help         Print this usage message and exit.\n"
              "  -f | --file         Save single frame to file and exit.\n"
              "  -n | --nopbo        Disable GL interop for display buffer.\n"
              "  -T | --tutorial-number <num>              Specify tutorial number\n"
              "  -t | --texture-path <path>                Specify path to texture directory\n"
              "App Keystrokes:\n"
              "  qVec  Quit\n"
              "  s  Save image to '" << SAMPLE_NAME << ".ppm'\n"
              << std::endl;

    exit(1);
}


int main( int argc, char** argv )
{
    std::string out_file;
    for( int i=1; i<argc; ++i )
    {
        const std::string arg( argv[i] );

        if( arg == "-h" || arg == "--help" )
        {
            printUsageAndExit( argv[0] );
        }
        else if ( arg == "-f" || arg == "--file" )
        {
            if( i == argc-1 )
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            out_file = argv[++i];
        }
        else if( arg == "-n" || arg == "--nopbo"  )
        {
            m_interop = false;
        }
        else if ( arg == "-t" || arg == "--texture-path" )
        {
            if ( i == argc-1 ) {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            texture_path = argv[++i];
        }
        else if ( arg == "-T" || arg == "--tutorial-number" )
        {
            if ( i == argc-1 ) {
                printUsageAndExit( argv[0] );
            }
            tutorial_number = atoi(argv[++i]);
            if ( tutorial_number < 0 || tutorial_number > 11 ) {
                std::cerr << "Tutorial number (" << tutorial_number << ") is out of range [0..11]\n";
                printUsageAndExit( argv[0] );
            }
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    if( texture_path.empty() ) {
        texture_path = std::string( sutil::samplesDir() ) + "/data";
    }

    try
    {
        glutInitialize( &argc, argv );

#ifndef __APPLE__
        glewInit();
#endif
        init();
        displayOnce();

        createContext();
        createGeometry();
        camera_eyeOld = make_float3( invOut[12], invOut[13], invOut[14] );
        setupCamera();
        setupLights();

        context->validate();

        if ( out_file.empty() )
        {
            glutRun();
        }
        else
        {
            updateCamera();
            context->launch( 0, width, height );
            sutil::displayBufferPPM( out_file.c_str(), getOutputBuffer() );
            destroyContext();
        }
        return 0;
    }
    SUTIL_CATCH( context->get() )
}


#pragma clang diagnostic pop
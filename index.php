<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <!-- The above 3 meta tags *must* come first in the head; any other head content must come *after* these tags -->
    <title>CS 4621: Final Project</title>

    <!-- Bootstrap -->
    <link href="css/bootstrap.min.css" rel="stylesheet">
    <link href="css/cs4620.css" rel="stylesheet">
    <link href="css/jquery-ui.min.css" rel="stylesheet">
    <link href="css/jquery-ui.theme.min.css" rel="stylesheet">
    <link href="css/jquery-ui.structure.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css?family=Acme|Kavivanar|Righteous" rel="stylesheet">

    <!-- HTML5 shim and Respond.js for IE8 support of HTML5 elements and media queries -->
    <!-- WARNING: Respond.js doesn't work if you view the page via file:// -->
    <!--[if lt IE 9]>
    <script src="https://oss.maxcdn.com/html5shiv/3.7.3/html5shiv.min.js"></script>
    <script src="https://oss.maxcdn.com/respond/1.4.2/respond.min.js"></script>
    <![endif]-->
</head>
<body>
  <style>
  body {
    font-family: 'Kavivanar', cursive;
    background: #333; color: rgb(229, 177, 58);
    margin: 20px;
  }
  h1 {
    text-align: center;
    font-family: 'Acme', sans-serif;
  }
  li {
    font-family: 'Righteous', cursive;
    list-style: none;
  }
</style>
<div class="container">
    <h1>CS 4621 Final Project</h1>

    <div align="center">
        <canvas id="webglCanvas" style="border: none; background-color: black;" width="600" height="600"></canvas>
    </div>

    <table class="table table-bordered">
        <tr>
            <td align="right">Render Mode</td>
            <td>
                <select id="renderMode" autocomplete="off">
                    <option value="POS_SPHERES">Position Spheres</option>
                    <option value="POS_SMOOTH">Smoothed Position Spheres</option>
                    <option value="POS_NORMALS">Normals</option>
                    <option value="THK_SPHERES">Thickness Spheres</option>
                    <option value="THK_SMOOTH">Smoothed Thickness Spheres</option>
                    <option value="FULL">Full Shading</option>
                </select>
            </td>
            <td align="right">Environment</td>
            <td>
                <input id="environment" type="checkbox">
            </td>
        </tr>
    </table>

    <h2>Controls</h2>
    <ul>
        <li>Click and drag to rotate the camera.</li>
        <li>Scroll to zoom in and out.</li>
    </ul>

    <br>

    <h2>Team Members</h2>

    <ul>
        <li>Runqian Huang (rh627)</li>
        <li>Sitian Chen (sc2294)</li>
        <li>Jason Liu (jl2732)</li>
        <li>Jiamin Zeng (jz863)</li>
    </ul>
</div>

<!-- jQuery (necessary for Bootstrap's JavaScript plugins) -->
<script src="js/jquery-3.1.1.min.js"></script>
<script src="js/jquery-ui.min.js"></script>
<script src="js/gl-matrix-min.js"></script>
<script src="js/preloadjs-0.6.2.min.js"></script>
<script src="//cdnjs.cloudflare.com/ajax/libs/highlight.js/9.9.0/highlight.min.js"></script>

<script id="quadVertex" type="vertex">
precision highp float;

attribute vec2 pos;

varying vec2 geom_texCoord;

void main(void) {
    geom_texCoord = (pos + 1.0) / 2.0;
    gl_Position = vec4(pos, 0.0, 1.0);
}
</script>

<script id="skyboxVertex" type="vertex">
precision highp float;

attribute vec3 coordinates;

uniform mat4 projection;
uniform mat4 view;

varying vec3 vRayDir;

void main() {
    mat4 noTransViewMatrix = mat4
        ( view[0].xyz, 0.0
        , view[1].xyz, 0.0
        , view[2].xyz, 0.0
        , 0.0, 0.0, 0.0, 1.0
        );
    gl_Position = projection * noTransViewMatrix * vec4(coordinates, 1.0);
    vRayDir = vec3(coordinates.x, coordinates.z, coordinates.y);
}
</script>

<script id="pointVertex" type="vertex">
/*
 *  in:  discrete points (for each fluid particle)
 * out: ``spheres''
 */
precision highp float;

attribute vec3 coordinates;

uniform float fov;
uniform float height;
uniform float radius;

uniform mat4 projection;
uniform mat4 view;

varying vec4 vEyePos;

void main(void) {
    vEyePos = view * vec4(coordinates, 1.0);
    vec3 ncenter = normalize(vEyePos.xyz);
    vec3 ntop = normalize(vEyePos.xyz + vec3(0.0, radius, 0.0));
    float angle = acos(dot(ncenter, ntop));
    float size = angle / fov * height;
    gl_Position = projection * vEyePos;
    gl_PointSize = size;
}
</script>

<script id="skyboxFragment" type="fragment">
precision highp float;

uniform samplerCube cubemap;

varying vec3 vRayDir;

void main() {
    gl_FragColor = vec4(textureCube(cubemap, normalize(vRayDir)).rgb, 1.0);
}
</script>

<script id="depthFragment" type="fragment">
/*
 *  in:  ``spheres''
 * out: eye space (x, y, z, depth)
 */
#extension GL_EXT_frag_depth : require
precision highp float;

uniform mat4 projection;
uniform float radius;

varying vec4 vEyePos;

void main(void) {
    vec3 n = vec3(2.0 * gl_PointCoord - 1.0, 0.0);
    n.y = -n.y; // PointCoord varies from 0 (top) to 1 (bottom)
    float r2sq = dot(n, n);
    if (r2sq > 1.0) discard;
    n.z = sqrt(1.0 - r2sq);
    vec4 pos = vec4(vEyePos.xyz + n * radius, 1.0);
    vec4 clip = projection * pos;
    gl_FragColor = vec4(pos.xyz, clip.z / clip.w);
    gl_FragDepthEXT = clip.z / clip.w;
}
</script>

<script id="thicknessFragment" type="fragment">
precision highp float;

uniform mat4 projection;
uniform float radius;

void main() {
    vec3 n = vec3(2.0 * gl_PointCoord - 1.0, 0.0);
    float r2sq = dot(n, n);
    if (r2sq > 1.0) discard;
    n.z = sqrt(1.0 - r2sq);

    float thickness = radius * 2. * n.z;
    gl_FragColor = vec4(thickness, thickness, thickness, 1.0);
}
</script>

<script id = "gaussianFragment" type = "fragment">
precision highp float;

const float blurRadius = 25.0; // smoothing radius

uniform vec2 blurDir;
uniform float blurScale; // sigma_d: spatial parameter
uniform float size; // side length of square canvas
uniform sampler2D texture;

varying vec2 geom_texCoord;

float blur() {
    float thickness = texture2D(texture, geom_texCoord).z;
    float sum = 0.;
    float wsum = 0.;

    for (float i = -blurRadius; i <= blurRadius; i += 1.0) {
        float sample = texture2D(texture, geom_texCoord + i / size * blurDir).z;
        float r = i * blurScale;
        float w = exp(-r*r);

        sum += sample * w;
        wsum += w;
    }

    if (wsum > 0.0) {
        sum = sum / wsum;
    }
    return sum;
}

void main() {
    float thickness = blur();
    gl_FragColor = vec4(thickness, thickness, thickness, 1.0);
}
</script>

<script id = "smoothFragment" type = "fragment">
precision highp float;

const float blurRadius = 25.0; // smoothing radius

uniform vec2 blurDir; // in direction (x, y); should be normalized
uniform float blurScale; // sigma_d: spatial parameter
uniform float blurDepthFalloff; // sigma_r: range parameter
uniform float fov;
uniform float size; // side length of square canvas
uniform sampler2D texture;

varying vec2 geom_texCoord;

float smooth() { // from "Screen Space Fluid Rendering for Games", NVIDIA
    float depth = texture2D(texture, geom_texCoord).z;
    if (depth > 0.0) { // -z is forwards. Anything with +z must be backwards
        discard;
    }
    float sum = 0.;
    float wsum = 0.;

    for (float i = -blurRadius; i <= blurRadius; i += 1.0) {
        float sample = texture2D(texture, geom_texCoord + i / size * blurDir).z;
        if (sample > 0.0) { //don't blur with background pixels
            continue;
        }
        // spatial domain
        float r = i * blurScale;
        float w = exp(-r*r);
        // range (depth) domain
        float r2 = (sample - depth) * blurDepthFalloff;
        float g = exp(-r2*r2);

        sum += sample * w * g;
        wsum += w * g;
    }

    if (wsum > 0.0) {
        sum = sum / wsum;
    }
    return sum;
}

vec3 getRay() { //getting ray in eyespace
    vec3 cameraEye = vec3(0., 0., 0.);
    vec3 cameraTarget = vec3(0., 0., -1.);
    vec3 cameraUp = vec3(0., 1., 0.);
    vec3 z = normalize(cameraEye-cameraTarget);
    vec3 x = normalize(cross(cameraUp, z));
    vec3 y = normalize(cross(z, x));
    float s = tan(fov / 2.);
    vec3 d = normalize(-1.0*z + s*(geom_texCoord.x*2. - 1.)*x + s*(geom_texCoord.y*2. - 1.)*y);
    return d;
}

vec3 recalculate(float smoothdepth) {
    vec3 p = vec3(0., 0., 0.);
    vec3 d = getRay();
    float t = smoothdepth/d.z; // t when the ray intersect at smoothed depth
    return t*d;
}

void main() {
    vec4 pos = texture2D(texture, geom_texCoord);
    float maxDepth = 0.; // when depth is not set by the texture, fragments in the background
    if (pos.z > maxDepth) {
        gl_FragColor = pos;
    } else {
        float depth = smooth();
        gl_FragColor = vec4(recalculate(depth), pos.w);
    }
}
</script>

<script id="normalFragment" type="fragment">
precision highp float;

varying vec2 geom_texCoord;

uniform sampler2D texture; // smoothed eyespace texture
uniform sampler2D thicknessTexture;
uniform samplerCube cubemap;

uniform float size;
uniform float r;
uniform vec3 L; // light direction
uniform vec3 attenuation; // Beer-Lambert attenuation coefficients (RGB)
uniform float roughness;
uniform vec3 lightColor;
uniform vec3 diffuseColor;
uniform bool normalOnly;
uniform bool useEnvmap;

vec3 calculateNormal() { // from "Screen Space Fluid Rendering for Games", NVIDIA
    vec3 eyePos = texture2D(texture, geom_texCoord).xyz;
    float maxDepth = 0.0;
    if (eyePos.z > maxDepth) {
        discard;
    }

    float texelSize = 1.0 / size;

    vec3 ddx = texture2D(texture, geom_texCoord + vec2(texelSize, 0.0)).xyz;
    float depthx1 = ddx.z;
    ddx -= eyePos;
    vec3 ddx2 = texture2D(texture, geom_texCoord + vec2(-texelSize, 0.0)).xyz;
    float depthx2 = ddx2.z;
    ddx2 = eyePos - ddx2;
    if (depthx1 > 0.0) {
        ddx = ddx2;
    } else if (depthx2 > 0.0) {
        ddx2 = ddx;
    }

    vec3 ddy = texture2D(texture, geom_texCoord + vec2(0.0, texelSize)).xyz;
    float depthy1 = ddy.z;
    ddy -= eyePos;
    vec3 ddy2 = texture2D(texture, geom_texCoord + vec2(0.0, -texelSize)).xyz;
    float depthy2 = ddy2.z;
    ddy2 = eyePos - ddy2;
    if (depthy1 > 0.0){
        ddy = ddy2;
    } else if (depthy2 > 0.0) {
        ddy2 = ddy;
    }

    return normalize(cross(ddx, ddy));
}

void main() {
    // Blinn-Phong
    vec3 N = calculateNormal();
    if (normalOnly) {
        gl_FragColor = vec4((N+1.0)/2.0, 1.0);
        return;
    }
    vec3 eyePos = texture2D(texture, geom_texCoord).xyz;
    vec3 V = normalize(-eyePos);
    vec3 H = normalize(L + V);
    vec3 R = 2.0*dot(N, V)*N - V;
    vec3 Ie = vec3(0.0);
    if (useEnvmap)
        Ie = textureCube(cubemap, R).rgb;
    else
        Ie = vec3(0.5); // constant gray environment light
    vec3 Is = vec3(1.0) * pow(max(dot(N, H), 0.0), 1.0 / roughness);
    float d = texture2D(thicknessTexture, geom_texCoord).z;
    vec3 Il = lightColor * exp(-attenuation * d);
    vec3 Id = diffuseColor * max(dot(N, L), 0.0);
    float a = 1.0 - exp(-0.3*d); // non-physical heuristic
    vec3 color = Il * (Ie + Id + Is) / (r * r);
    if (length(color) >= 1.7) { // preserve highlights; sqrt(3) = 1.73
        a = 1.0;
    }
    gl_FragColor = vec4(color, a);
}
</script>

<script id="copyFragment" type="fragment">
precision highp float;

uniform sampler2D texture;

varying vec2 geom_texCoord;

void main() {
    gl_FragColor = texture2D(texture, geom_texCoord);
}
</script>

<script>
const size = 600;
const radius = 0.1;

function rad(degrees) {
    return degrees * Math.PI / 180.0;
}

const canvas = $("#webglCanvas");
function addCameraControls(gl, projCallback, viewCallback, fovCallback) {
    const canvasRect = canvas[0].getBoundingClientRect();
    const EPS = 1e-4;
    const minfov = rad(20);
    const maxfov = rad(80);
    let origin = [0.5, 0.5, 0.5];
    let pa = [0, rad(90)]; // angles: polar, azimuth
    let fov = rad(60);
    const up = vec3.fromValues(0, 0, 1);

    let proj = mat4.create();
    function updateProj() {
        mat4.perspective(proj, fov, 1, 0.1, 100);
        projCallback(proj);
    }

    let view = mat4.create();
    let eye = vec3.create();
    let target = vec3.create();
    let offset = vec3.create();
    function updateView() {
        vec3.set(eye, ...origin);
        vec3.set(offset
            , Math.sin(pa[1])*Math.cos(pa[0])
            , Math.sin(pa[1])*Math.sin(pa[0])
            , Math.cos(pa[1])
            );
        vec3.add(eye, eye, offset);
        vec3.set(target, ...origin);
        mat4.lookAt(view, eye, target, up);
        viewCallback(view);
    }

    let moveCamera = false;
    let prev;
    canvas.mousedown(e => moveCamera = true);
    canvas.mouseup(e => moveCamera = false);
    canvas.mousemove(e => {
        if (moveCamera) {
            pa[0] += (e.clientX - prev.clientX) / canvasRect.width * fov;
            pa[1] += (e.clientY - prev.clientY) / canvasRect.height * fov;

            while (pa[0] > Math.PI)
                pa[0] -= 2*Math.PI;
            while (pa[0] < -Math.PI)
                pa[0] += 2*Math.PI;

            if (pa[1] > Math.PI - EPS)
                pa[1] = Math.PI - EPS;
            else if (pa[1] < EPS)
                pa[1] = EPS;
            updateView();
        }
        prev = e;
    });
    canvas[0].addEventListener("wheel", e => {
        if (e.deltaY > 0)
            fov += rad(2);
        else
            fov -= rad(2);
        if (fov > maxfov)
            fov = maxfov;
        else if (fov < minfov)
            fov = minfov;
        e.preventDefault();
        updateProj();
        fovCallback(fov);
    });
    updateProj();
    updateView();
    fovCallback(fov);
}

function initializeWebGL() {
    let gl = null;
    try {
        gl = canvas[0].getContext("experimental-webgl");
        if (!gl) {
            gl = canvas[0].getContext("webgl");
        }
    } catch (error) {}
    if (!gl) {
        alert("Could not get WebGL context!");
        throw new Error("Could not get WebGL context!");
    }
    return gl;
}

function createShader(gl, shaderScriptId) {
    let shaderScript = $("#" + shaderScriptId);
    let shaderSource = shaderScript[0].text;
    let shaderType = null;
    if (shaderScript[0].type == "vertex") {
        shaderType = gl.VERTEX_SHADER;
    } else if (shaderScript[0].type == "fragment") {
        shaderType = gl.FRAGMENT_SHADER;
    } else {
        throw new Error("Invalid shader type: " + shaderScript[0].type)
    }
    let shader = gl.createShader(shaderType);
    gl.shaderSource(shader, shaderSource);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        let infoLog = gl.getShaderInfoLog(shader);
        gl.deleteShader(shader);
        throw new Error("An error occurred compiling the shader: " + infoLog);
    } else {
        return shader;
    }
}

function createGlslProgram(gl, vertexShaderId, fragmentShaderId) {
    let program = gl.createProgram();
    gl.attachShader(program, createShader(gl, vertexShaderId));
    gl.attachShader(program, createShader(gl, fragmentShaderId));
    gl.linkProgram(program);
    gl.validateProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        let infoLog = gl.getProgramInfoLog(program);
        gl.deleteProgram(program);
        throw new Error("An error occurred linking the program: " + infoLog);
    } else {
        return program;
    }
}

function createShape(gl, vertexData, indexData) {
    var shape = {};

    var vertexArray = new Float32Array(vertexData);
    var vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertexArray, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    var indexArray = new Uint16Array(indexData);
    var indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indexArray, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

    shape.vertexBuffer = vertexBuffer;
    shape.indexBuffer = indexBuffer;
    shape.size = indexData.length;

    return shape;
}

function createFloatTexture(gl, type, width, height) {
    let texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, type, width, height, 0, type, gl.FLOAT, null);
    for (const s of ["MAG", "MIN"]) {
        gl.texParameteri(gl.TEXTURE_2D, gl["TEXTURE_" + s + "_FILTER"], gl.NEAREST);
    }
    for (const s of ["S", "T"]) {
        gl.texParameteri(gl.TEXTURE_2D, gl["TEXTURE_WRAP_" + s], gl.CLAMP_TO_EDGE);
    }
    gl.bindTexture(gl.TEXTURE_2D, null);
    return texture;
}

function createPoints(gl, vertexData, indexData) {
    let shape = {};
    let vertexArray = new Float32Array(vertexData);
    let vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertexArray, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    let indexArray = new Uint16Array(indexData);
    let indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indexArray, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    shape.vertexBuffer = vertexBuffer;
    shape.indexBuffer = indexBuffer;
    shape.size = indexData.length;
    return shape;
}

function drawPoints(gl, prog, shape) {
    gl.bindBuffer(gl.ARRAY_BUFFER, shape.vertexBuffer);
    gl.enableVertexAttribArray(prog.coordinates);
    gl.vertexAttribPointer(prog.coordinates, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, shape.indexBuffer);
    gl.drawElements(gl.POINTS, shape.size, gl.UNSIGNED_SHORT, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    gl.useProgram(null);
}

function makeFullQuad(gl) {
    let vs = new Float32Array(
        [-1.0,-1.0
        , 1.0,-1.0
        ,-1.0, 1.0
        , 1.0, 1.0
        ]);
    let vBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vs, gl.STATIC_DRAW);

    let is = new Uint16Array([0, 1, 2, 3]);
    let iBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, iBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, is, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

    return function drawQuad(prog, texture) {
        gl.useProgram(prog);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, texture);

        gl.bindBuffer(gl.ARRAY_BUFFER, vBuffer);
        gl.enableVertexAttribArray(prog.pos);
        gl.vertexAttribPointer(prog.pos, 2, gl.FLOAT, false, 2*4, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);

        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, iBuffer);
        gl.drawElements(gl.TRIANGLE_STRIP, is.length, gl.UNSIGNED_SHORT, 0);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

        gl.bindTexture(gl.TEXTURE_2D, null);
        gl.useProgram(null);
    }
}

function makeSkyboxTexture(gl, prog, queue) {
    gl.useProgram(prog);
    let texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_CUBE_MAP, texture);
    for (const d of ["X", "Y", "Z"]) {
        for (const s of ["POSITIVE_" + d, "NEGATIVE_" + d]) {
            gl.texImage2D(gl["TEXTURE_CUBE_MAP_" + s], 0, gl.RGBA, gl.RGBA,
                gl.UNSIGNED_BYTE, queue.getResult(s, false));
        }
    }
    gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    for (const s of ["S", "T"]) {
        gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl["TEXTURE_WRAP_" + s], gl.REPEAT);
    }
    gl.uniform1i(prog.cubemap, 0);
    gl.bindTexture(gl.TEXTURE_CUBE_MAP, null);
    gl.useProgram(null);
    return texture;
}

function makeSkybox(gl, prog, texture) {
    let cube = {};
    cube.vertices = new Float32Array(
        [-10,-10,-10
        ,-10, 10,-10
        , 10,-10,-10
        , 10, 10,-10
        ,-10,-10, 10
        ,-10, 10, 10
        , 10,-10, 10
        , 10, 10, 10
        ]);
    cube.indices = new Uint16Array(
        [ 0, 1, 2
        , 1, 2, 3
        , 0, 4, 1
        , 4, 1, 5
        , 0, 2, 4
        , 2, 4, 6
        , 7, 6, 3
        , 6, 3, 2
        , 7, 5, 3
        , 5, 3, 1
        , 7, 5, 6
        , 5, 6, 4
        ]);
    cube.vbuffer = gl.createBuffer();
    cube.ibuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, cube.vbuffer);
    gl.bufferData(gl.ARRAY_BUFFER, cube.vertices, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, cube.ibuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, cube.indices, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

    return function drawSkybox() {
        gl.useProgram(prog);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_CUBE_MAP, texture);

        gl.bindBuffer(gl.ARRAY_BUFFER, cube.vbuffer);
        gl.vertexAttribPointer(prog.position, 3, gl.FLOAT, false, 3*4, 0);
        gl.enableVertexAttribArray(prog.position);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, cube.ibuffer);
        gl.drawElements(gl.TRIANGLES, cube.indices.length, gl.UNSIGNED_SHORT, 0);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
        gl.bindTexture(gl.TEXTURE_CUBE_MAP, null);
        gl.useProgram(null);
    }
}

function makeDoubleBuffer(gl, type, width, height) {
    let fbo = gl.createFramebuffer();
    let textures =
        [ createFloatTexture(gl, type, width, height)
        , createFloatTexture(gl, type, width, height)
        ];
    let readIndex = 0;

    function write(draw) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
            gl.TEXTURE_2D, textures[1 - readIndex], 0);

        draw();

        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
            gl.TEXTURE_2D, null, 0);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        readIndex = 1 - readIndex;
    }

    return { write: write, read: () => textures[readIndex] };
}

/* BEGIN PROGRAMS */
function addLocations(gl, prog, attribs, unifs) {
    for (const s of attribs) {
        prog[s] = gl.getAttribLocation(prog, s);
    }
    for (const s of unifs) {
        prog[s] = gl.getUniformLocation(prog, s);
    }
}

function makeDepthProgram(gl) {
    let prog = createGlslProgram(gl, "pointVertex", "depthFragment");
    let attribs =
        [ "coordinates"
        ];
    let unifs =
        [ "projection"
        , "view"
        , "radius"
        , "fov"
        , "height"
        ];
    addLocations(gl, prog, attribs, unifs);

    // known constants
    gl.useProgram(prog);
    gl.uniform1f(prog.height, size);
    gl.useProgram(null);

    return prog;
}

function makeSmoothProgram(gl) {
    let prog = createGlslProgram(gl, "quadVertex", "smoothFragment");
    let attribs =
        [ "pos"
        ];
    let unifs =
        [ "blurDir"
        , "blurScale"
        , "blurDepthFalloff"
        , "fov"
        , "size"
        , "texture"
        ];
    addLocations(gl, prog, attribs, unifs);

    // known constants
    gl.useProgram(prog);
    gl.uniform1f(prog.blurScale, 0.1);
    gl.uniform1f(prog.blurDepthFalloff, 0.5);
    gl.uniform1f(prog.size, size);
    gl.uniform1i(prog.texture, 0);
    gl.useProgram(null);

    return prog;
}

function makeThicknessProgram(gl) {
    let prog = createGlslProgram(gl, "pointVertex", "thicknessFragment");
    let attribs =
        [ "coordinates"
        ];
    let unifs =
        [ "projection"
        , "view"
        , "radius"
        , "near"
        , "fov"
        , "height"
        ];
    addLocations(gl, prog, attribs, unifs);

    // known constants
    gl.useProgram(prog);
    gl.uniform1f(prog.near, 0.1);
    gl.uniform1f(prog.height, size);
    gl.useProgram(null);

    return prog;
}

function makeGaussianProgram(gl) {
    let prog = createGlslProgram(gl, "quadVertex", "gaussianFragment");
    let attribs =
        [ "pos"
        ];
    let unifs =
        [ "blurDir"
        , "blurScale"
        , "size"
        , "texture"
        ];
    addLocations(gl, prog, attribs, unifs);

    // known constants
    gl.useProgram(prog);
    gl.uniform1f(prog.blurScale, 0.1);
    gl.uniform1f(prog.size, size);
    gl.uniform1i(prog.texture, 0);
    gl.useProgram(null);

    return prog;
}

function makeNormalProgram(gl) {
    let prog = createGlslProgram(gl, "quadVertex", "normalFragment");
    let attribs =
        [ "pos"
        ];
    let unifs =
        [ "r"
        , "L"
        , "attenuation"
        , "roughness"
        , "lightColor"
        , "diffuseColor"
        , "size"
        , "texture"
        , "thicknessTexture"
        , "normalOnly"
        , "useEnvmap"
        , "cubemap"
        ];
    addLocations(gl, prog, attribs, unifs);

    // known constants
    gl.useProgram(prog);
    gl.uniform1f(prog.r, 1.0);
    gl.uniform3f(prog.L, 0.0, 0.8, -0.6);
    gl.uniform3f(prog.attenuation, 0.6, 0.2, 0.01);
    gl.uniform1f(prog.roughness, 0.02);
    gl.uniform3f(prog.lightColor, 1.0, 1.0, 1.0);
    gl.uniform3f(prog.diffuseColor, 0.6, 0.7, 1.0);
    gl.uniform1f(prog.size, size);
    gl.uniform1i(prog.texture, 0);
    gl.uniform1i(prog.thicknessTexture, 1);
    gl.uniform1i(prog.cubemap, 2);
    gl.uniform1i(prog.normalOnly, 0);
    gl.useProgram(null);

    // add listener to checkbox
    const envCheckbox = $("#environment");
    envCheckbox.change(e => {
        gl.useProgram(prog);
        if (e.target.checked)
            gl.uniform1i(prog.useEnvmap, 1);
        else {
            gl.uniform1i(prog.useEnvmap, 0);
        }
        gl.useProgram(null);
    });
    envCheckbox.change();

    return prog;
}

function makeCopyProgram(gl) {
    let prog = createGlslProgram(gl, "quadVertex", "copyFragment");
    let attribs =
        [ "pos"
        ];
    let unifs =
        [ "texture"
        ];
    addLocations(gl, prog, attribs, unifs);

    // known constants
    gl.useProgram(prog);
    gl.uniform1i(prog.texture, 0);
    gl.useProgram(null);

    return prog;
}

function makeSkyboxProgram(gl) {
  var prog = createGlslProgram(gl, "skyboxVertex", "skyboxFragment");
  let attribs =
      [ "coordinates"
      ];
  let unifs =
      [ "projection"
      , "view"
      ];
  addLocations(gl, prog, attribs, unifs);
  gl.useProgram(prog);

  gl.useProgram(null);
  return prog;
}
/* END PROGRAMS */

/* BEGIN SIMULATION */
function time2stamp(){
    var d = new Date();
    return Date.parse(d) + d.getMilliseconds();
}
function empty(array){ //empty an array
    array.length = 0;
}
function Poly_kernel(pi, pj, h){ //Poly6 kernel for density estimation
    var r_2 = Math.pow(pi.newposition[0] - pj.newposition[0], 2.0) + Math.pow(pi.newposition[1] - pj.newposition[1], 2.0)
        + Math.pow(pi.newposition[2] - pj.newposition[2], 2.0);
    if (r_2 > (h * h)){return 0;}
    var t = h * h - r_2;
    var s = 315.0 / (64.0 * 3.14159265359 * Math.pow(h, 9.0));
    return s * Math.pow(t, 3.0);
}
function W(r, h){ //Poly6 kernel for density estimation
    if (r > h){return 0;}
    var t = h * h - r * r;
    var s = 315.0 / (64.0 * 3.14159265359 * Math.pow(h, 9.0));
    return s * Math.pow(t, 3.0);
}
function delta_W(pi, pj, h){ //gradient of Poly6 kernel
    var r_2 = Math.pow(pi.newposition[0] - pj.newposition[0], 2.0) + Math.pow(pi.newposition[1] - pj.newposition[1], 2.0)
        + Math.pow(pi.newposition[2] - pj.newposition[2], 2.0);
    var r_ = [0, 0, 0];
    r_[0] = pi.newposition[0] - pj.newposition[0];
    r_[1] = pi.newposition[1] - pj.newposition[1];
    r_[2] = pi.newposition[2] - pj.newposition[2];
    var t = h * h - r_2;
    var s = 945.0 / (32.0 * 3.14159265359 * Math.pow(h, 9.0));
    r_[0] = -r_[0] * s * t * t;
    r_[1] = -r_[1] * s * t * t;
    r_[2] = -r_[2] * s * t * t;
    return r_;
}
function Spiky_kernel(pi, pj, h){ //Spiky kernel for gradient calculation
    var r = Math.sqrt(Math.pow(pi.newposition[0] - pj.newposition[0], 2.0) + Math.pow(pi.newposition[1] - pj.newposition[1], 2.0)
        + Math.pow(pi.newposition[2] - pj.newposition[2], 2.0));
    var r_ = [0, 0, 0];
    r_[0] = pi.newposition[0] - pj.newposition[0];
    r_[1] = pi.newposition[1] - pj.newposition[1];
    r_[2] = pi.newposition[2] - pj.newposition[2];
    var s = 45.0 / (3.14159265359 * Math.pow(h, 6.0));
    var t = h - r;
    if (r != 0){
        r_[0] = r_[0] * s * t * t / r;
        r_[1] = r_[1] * s * t * t / r;
        r_[2] = r_[2] * s * t * t / r;
    }
    else{
        r_[0] = r_[0] * s * t * t;
        r_[1] = r_[1] * s * t * t;
        r_[2] = r_[2] * s * t * t;
    }
    return r_;
}
function rho(i, water, h){ //function pi = sum(mj * W(pi - pj, h)), and mj is always 1 here
    var mass = 3;
    var numofneighbor = water[i].neighbor.length;
    var sum = 0;
    for (var j = 0; j < numofneighbor; j++){
        sum += mass * Poly_kernel(water[i], water[i].neighbor[j], h);
    }
    return sum;
}
function C(i, water, h, rho0){ //function Ci(p1, ... , pn), and p0 is set to be 1000
    return (rho(i, water, h) / rho0 - 1);
}
function sum_len_delta_C(i, water, h, rho0){ //function sum((gradient of Ci with respect to particle k)^2)
    var numofneighbor = water[i].neighbor.length;
    var sum = [0, 0, 0];
    for (var j = 0; j < numofneighbor; j++){
        var result = delta_W(water[i], water[i].neighbor[j], h);
        sum[0] += result[0];
        sum[1] += result[1];
        sum[2] += result[2];
    }
    sum[0] = sum[0] / rho0;
    sum[1] = sum[1] / rho0;
    sum[2] = sum[2] / rho0;
    var final = sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2];
    for (var j = 0; j < numofneighbor; j++){
        var result = delta_W(water[i], water[i].neighbor[j], h);
        result[0] = result[0] / rho0;
        result[1] = result[1] / rho0;
        result[2] = result[2] / rho0;
        final += (result[0] * result[0] + result[1] * result[1] + result[2] * result[2]);
    }
    return final;
}
function lameta(i, water, h, e, rho0){ //function lametai = - Ci / (sum_len_delta_C + e)
    var c = C(i, water, h, rho0);
    var delta_c = sum_len_delta_C(i, water, h, rho0);
    return (- c / (delta_c + e));
}
function scorr(pi, pj, h, k, n, delta_q){ //tensile instability
    var t = Poly_kernel(pi, pj, h);
    var t1 = W(delta_q, h);
    var result = - k * Math.pow(t / t1, n);
    return result;
}
function delta_p(i, water, h, e, k, n, delta_q, rho0){ //position update
    var numofneighbor = water[i].neighbor.length;
    var sum = [0, 0, 0];
    for (var j = 0; j < numofneighbor; j++){
        var x = water[i].neighbor[j].id;
        var lametai = lameta(i, water, h, e, rho0);
        var lametaj = lameta(x, water, h, e, rho0);
        var scorr_ = scorr(water[i], water[i].neighbor[j], h, k, n, delta_q);
        var deltaW = Spiky_kernel(water[i], water[i].neighbor[j], h);
        for (let index = 0; index < 3; index++) {
            sum[index] += (lametai + lametaj + scorr_) * deltaW[index];
            //sum[index] += (lametai + lametaj) * deltaW[index];
        }
    }
    sum.forEach(function(element){
        element /= rho0;
    })
    return sum;
}
function omega(i, water, h){//calculate vorticity
    var result = [0, 0, 0];
    var numofneighbor = water[i].neighbor.length;
    for (var j = 0; j < numofneighbor; j++){
        var v = [0, 0, 0];
        v[0] = water[i].neighbor[j].velocity[0] - water[i].velocity[0];
        v[1] = water[i].neighbor[j].velocity[1] - water[i].velocity[1];
        v[2] = water[i].neighbor[j].velocity[2] - water[i].velocity[2];
        var w = delta_W(water[i], water[i].neighbor[j], h);
        w[0] = -w[0];
        w[1] = -w[1];
        w[2] = -w[2];
        var cross = [0, 0, 0];
        cross[0] = v[1] * w[2] - v[2] * w[1];
        cross[1] = v[2] * w[0] - v[0] * w[2];
        cross[2] = v[0] * w[1] - v[1] * w[0];
        for (let index = 0; index < 3; index++) {
            result[index] += cross[index];
        }
    }
    water[i].vorticity[0] = result[0];
    water[i].vorticity[1] = result[1];
    water[i].vorticity[2] = result[2];
}
function corr_force(i, water, h, rho0, e){//calculate corrective force of vorticity
    var result = [0, 0, 0];
    var numofneighbor = water[i].neighbor.length;
    for (var j = 0; j < numofneighbor; j++){
        var len = Math.pow(water[i].neighbor[j].vorticity[0], 2) + Math.pow(water[i].neighbor[j].vorticity[1], 2)
            + Math.pow(water[i].neighbor[j].vorticity[2], 2);
        var abs_omega = Math.sqrt(len);
        var w = delta_W(water[i], water[i].neighbor[j], h);
        w[0] = -w[0];
        w[1] = -w[1];
        w[2] = -w[2];
        var temp = [0, 0, 0];
        temp[0] = w[0] * abs_omega / rho0;
        temp[1] = w[1] * abs_omega / rho0;
        temp[2] = w[2] * abs_omega / rho0;
        for (let index = 0; index < 3; index++) {
            result[index] += temp[index];
        }
    }
    var len_result = Math.pow(result[0], 2) + Math.pow(result[1], 2) + Math.pow(result[2], 2);
    var N = [0, 0, 0];
    if (len_result != 0){
        N[0] = result[0] / Math.sqrt(len_result);
        N[1] = result[1] / Math.sqrt(len_result);
        N[2] = result[2] / Math.sqrt(len_result);
    }
    var t = [0, 0, 0];
    var ome = water[i].vorticity;
    t[0] = N[1] * ome[2] - N[2] * ome[1];
    t[1] = N[2] * ome[0] - N[0] * ome[2];
    t[2] = N[0] * ome[1] - N[1] * ome[0];
    var final = [0, 0, 0];
    final[0] = t[0] * e;
    final[1] = t[1] * e;
    final[2] = t[2] * e;
    return final;
}
function cor_v(i, water, c, h){
    var result = [0, 0, 0];
    var numofneighbor = water[i].neighbor.length;
    for (var j = 0; j < numofneighbor; j++){
        var w = Poly_kernel(water[i], water[i].neighbor[j], h);
        var v = [0, 0, 0];
        v[0] = water[i].neighbor[j].velocity[0] - water[i].velocity[0];
        v[1] = water[i].neighbor[j].velocity[1] - water[i].velocity[1];
        v[2] = water[i].neighbor[j].velocity[2] - water[i].velocity[2];
        var temp = [0, 0, 0];
        temp[0] = v[0] * w * c;
        temp[1] = v[1] * w * c;
        temp[2] = v[2] * w * c;
        for (let index = 0; index < 3; index++) {
            result[index] += temp[index];
        }
    }
    return result;
}
function set_grid(water, grid){ //arrange particles into the grids
    var water_size = water.length;
    var grid_size = grid.length;
    for (var i = 0; i < water_size; i++){
        water[i].gridnum = -1; //reset
    }
    for (var i = 0; i < grid_size; i++){
        empty(grid[i].particles); //reset
        for (var j = 0; j < water_size; j++){
            if (water[j].gridnum == -1){ //if the particle has not been put into a grid
                if((water[j].newposition[0] >= grid[i].xlo)&&(water[j].newposition[0] <= grid[i].xhi) //if the particle is in the grid
                    &&(water[j].newposition[1] >= grid[i].ylo)&&(water[j].newposition[1] <= grid[i].yhi)
                    &&(water[j].newposition[2] >= grid[i].zlo)&&(water[j].newposition[2] <= grid[i].zhi)){
                    grid[i].particles.push(water[j]);
                    water[j].gridnum = i;
                }
            }
        }
    }
}
function isneighbor(g, t, condition){ //determine whether two particles are neighbors
    var g_size = g.particles.length;
    for (var j = 0; j < g_size; j++){
        var d = Math.pow(g.particles[j].newposition[0] - t.newposition[0], 2.0) + Math.pow(g.particles[j].newposition[1] - t.newposition[1], 2.0)
            + Math.pow(g.particles[j].newposition[2] - t.newposition[2], 2.0);
        if ((d <= Math.pow(condition, 2.0))&&(d != 0)){
            t.neighbor.push(g.particles[j]);
        }
    }
}
function find_neighbor(water, grid, condition){ //find the neighbors of each particle
    var water_size = water.length;
    var grid_size = grid.length;
    for (var i = 0; i < water_size; i++){
        empty(water[i].neighbor); //reset
        var g = grid[water[i].gridnum]; //check the neighbor 27 grids, including itself
        isneighbor(g, water[i], condition);
        if (g.isxedgel == false){
            var t = grid[water[i].gridnum - 1];
            isneighbor(t, water[i], condition);
            if (g.isyedgel == false){
                t = grid[water[i].gridnum - 1 - g.l];
                isneighbor(t, water[i], condition);
                if (g.iszedgel == false){
                    t = grid[water[i].gridnum - 1 - g.l * g.w - g.l];
                    isneighbor(t, water[i], condition);
                }
                if (g.iszedger == false){
                    t = grid[water[i].gridnum - 1 + g.l * g.w - g.l];
                    isneighbor(t, water[i], condition);
                }
            }
            if (g.isyedger == false){
                t = grid[water[i].gridnum - 1 + g.l];
                isneighbor(t, water[i], condition);
                if (g.iszedgel == false){
                    t = grid[water[i].gridnum - 1 - g.l * g.w + g.l];
                    isneighbor(t, water[i], condition);
                }
                if (g.iszedger == false){
                    t = grid[water[i].gridnum - 1 + g.l * g.w + g.l];
                    isneighbor(t, water[i], condition);
                }
            }
            if (g.iszedgel == false){
                t = grid[water[i].gridnum - 1 - g.l * g.w];
                isneighbor(t, water[i], condition);
            }
            if (g.iszedger == false){
                t = grid[water[i].gridnum - 1 + g.l * g.w];
                isneighbor(t, water[i], condition);
            }
        }
        if (g.isxedger == false){
            var t = grid[water[i].gridnum + 1];
            isneighbor(t, water[i], condition);
            if (g.isyedgel == false){
                t = grid[water[i].gridnum + 1 - g.l];
                isneighbor(t, water[i], condition);
                if (g.iszedgel == false){
                    t = grid[water[i].gridnum + 1 - g.l * g.w - g.l];
                    isneighbor(t, water[i], condition);
                }
                if (g.iszedger == false){
                    t = grid[water[i].gridnum + 1 + g.l * g.w - g.l];
                    isneighbor(t, water[i], condition);
                }
            }
            if (g.isyedger == false){
                t = grid[water[i].gridnum + 1 + g.l];
                isneighbor(t, water[i], condition);
                if (g.iszedgel == false){
                    t = grid[water[i].gridnum + 1 - g.l * g.w + g.l];
                    isneighbor(t, water[i], condition);
                }
                if (g.iszedger == false){
                    t = grid[water[i].gridnum + 1 + g.l * g.w + g.l];
                    isneighbor(t, water[i], condition);
                }
            }
            if (g.iszedgel == false){
                t = grid[water[i].gridnum + 1 - g.l * g.w];
                isneighbor(t, water[i], condition);
            }
            if (g.iszedger == false){
                t = grid[water[i].gridnum + 1 + g.l * g.w];
                isneighbor(t, water[i], condition);
            }
        }
        if (g.isyedgel == false){
            var t = grid[water[i].gridnum - g.l];
            isneighbor(t, water[i], condition);
            if (g.iszedgel == false){
                t = grid[water[i].gridnum - g.l - g.l * g.w];
                isneighbor(t, water[i], condition);
            }
            if (g.iszedger == false){
                t = grid[water[i].gridnum - g.l + g.l * g.w];
                isneighbor(t, water[i], condition);
            }
        }
        if (g.isyedger == false){
            var t = grid[water[i].gridnum + g.l];
            isneighbor(t, water[i], condition);
            if (g.iszedgel == false){
                t = grid[water[i].gridnum + g.l - g.l * g.w];
                isneighbor(t, water[i], condition);
            }
            if (g.iszedger == false){
                t = grid[water[i].gridnum + g.l + g.l * g.w];
                isneighbor(t, water[i], condition);
            }
        }
        if (g.iszedgel == false){
            var t = grid[water[i].gridnum - g.l * g.w];
            isneighbor(t, water[i], condition);
        }
        if (g.iszedger == false){
            var t = grid[water[i].gridnum + g.l * g.w];
            isneighbor(t, water[i], condition);
        }
    }
}

function runCanvas(queue) {
    let gl = initializeWebGL();
    gl.getExtension("OES_texture_float");
    gl.getExtension("EXT_frag_depth");

    const GRAVITY = [0.0, 0.0, -9.8];//9.8
    var numofparticles = 1000; // number of particles//1000
    var length = 10; //limitation of the space in integer//10
    var width = 10;//10
    var height = 40;//40
    var height_ = 20;
    var gain = 20.0; //the gain used to transform the integer limitation into float coordinates//20.0
    var deltat = 0.00075; //the time step of refreshing the canvas//0.00055
    var sizeofgrid = 6; //size of a single grid//6
    var n_condition = 1; //condition of neighbors//1
    var h = 0.2;//0.02
    var e = 0.9989;//0.9991
    var e1 = 0.9;
    var delta_q = 0.08 * h;//0.08
    var k = 0.1;//0.1
    var n = 4;//4
    var rho0 = 1300.0;//1300
    var c = 0.01;
    var condition = n_condition / gain;
    var water = []; //store the particles
    for (var i = 0; i<numofparticles; i++) {
        var end = water.length; //Add new node to the end of the array
        water[end] = new Object();
        var x = i % length; //initial positions
        var y = ((i - (i % length)) / length) % width;
        var z = (i - i % (length * width)) / (length * width);
        if ((z + 2 * y - 2 * (width - 1)) > 0){
            y = width - 1 - y;
            z = 2 * (numofparticles / (width * length)) - z;
        }
        x = x / gain;
        y = y / gain;
        z = z / gain;
        water[end].id = i;
        water[end].oldposition = [x, y, z];
        water[end].newposition = [x, y, z];
        water[end].velocity = [0.0, 0.0, 0.0]; //initial velocities
        water[end].vorticity = [0.0, 0.0, 0.0]; //initial vorticity
        water[end].gridnum = -1;
        water[end].neighbor = [];
    }
    var grid = []; //grid to help find neighbors
    var l; //the number of grids in x-direction
    var lo; //the average length that is outside of the space limitation in x-direction
    var w; //the number of grids in y-direction
    var wo; //the average length that is outside of the space limitation in y-direction
    var h; //the number of grids in z-direction
    var ho; //the average length that is outside of the space limitation in z-direction
    if (length % sizeofgrid != 0){
        l = (length - length % sizeofgrid) / sizeofgrid + 1;
        lo = (l * sizeofgrid - length) / 2.0;
    }
    else{
        l = length / sizeofgrid;
        lo = 0;
    }
    if (width % sizeofgrid != 0){
        w = (width - width % sizeofgrid) / sizeofgrid + 1;
        wo = (w * sizeofgrid - width) / 2.0;
    }
    else{
        w = width / sizeofgrid;
        wo = 0;
    }
    if (height % sizeofgrid != 0){
        h = (height - height % sizeofgrid) / sizeofgrid + 1;
        ho = (h * sizeofgrid - height) / 2.0;
    }
    else{
        h = height / sizeofgrid;
        ho = 0;
    }
    for (var i = 0; i < h; i++){
        for (var j = 0; j < w; j++){
            for (var k = 0; k < l; k++){
                var end = grid.length; //Add new node to the end of the array
                grid[end] = new Object();
                grid[end].xlo = k * sizeofgrid - lo - 0.5; //index range of this grid
                grid[end].xhi = grid[end].xlo + sizeofgrid;
                grid[end].ylo = j * sizeofgrid - wo - 0.5;
                grid[end].yhi = grid[end].ylo + sizeofgrid;
                grid[end].zlo = i * sizeofgrid - ho - 0.5;
                grid[end].zhi = grid[end].zlo + sizeofgrid;
                grid[end].xlo /= gain; //coordinates range of this grid
                grid[end].xhi /= gain;
                grid[end].ylo /= gain;
                grid[end].yhi /= gain;
                grid[end].zlo /= gain;
                grid[end].zhi /= gain;
                grid[end].l = l; //size of the whole grid map
                grid[end].w = w;
                grid[end].h = h;
                grid[end].isxedgel = false; //if this grid is on the edge of x
                grid[end].isxedger = false;
                grid[end].isyedgel = false; //if this grid is on the edge of y
                grid[end].isyedger = false;
                grid[end].iszedgel = false; //if this grid is on the edge of z
                grid[end].iszedger = false;
                if (i == 0){grid[end].iszedgel = true;}
                if (i == (h - 1)){grid[end].iszedger = true;}
                if (j == 0){grid[end].isyedgel = true;}
                if (j == (w - 1)){grid[end].isyedger = true;}
                if (k == 0){grid[end].isxedgel = true;}
                if (k == (l - 1)){grid[end].isxedger = true;}
                grid[end].particles = [];
            }
        }
    }
    set_grid(water, grid);
    find_neighbor(water, grid, condition);
/* END SIMULATION */

    let depthProg = makeDepthProgram(gl);
    let smoothProg = makeSmoothProgram(gl);
    let normalProg = makeNormalProgram(gl);
    let thicknessProg = makeThicknessProgram(gl);
    let gaussianProg = makeGaussianProgram(gl);
    let copyProg = makeCopyProgram(gl);
    let skyboxProg = makeSkyboxProgram(gl);

    let vertices = [];
    let indices = [];
    let pts = createPoints(gl, vertices, indices);

    let img = makeDoubleBuffer(gl, gl.RGBA, size, size);
    let thickBuffer = makeDoubleBuffer(gl, gl.RGBA, size, size);
    let drawQuad = makeFullQuad(gl);
    let skyboxTexture = makeSkyboxTexture(gl, skyboxProg, queue);
    let drawSkybox = makeSkybox(gl, skyboxProg, skyboxTexture);

    addCameraControls(gl
        , proj => {
            for (const p of [depthProg, thicknessProg, skyboxProg]) {
                gl.useProgram(p);
                gl.uniformMatrix4fv(p.projection, false, proj);
                gl.useProgram(null);
            }
        }
        , view => {
            for (const p of [depthProg, thicknessProg, skyboxProg]) {
                gl.useProgram(p);
                gl.uniformMatrix4fv(p.view, false, view);
                gl.useProgram(null);
            }
        }
        , fov => {
            for (const p of [depthProg, smoothProg, thicknessProg]) {
                gl.useProgram(p);
                gl.uniform1f(p.fov, fov);
                gl.useProgram(null);
            }
            for (const p of [depthProg, thicknessProg]) {
                gl.useProgram(p);
                gl.uniform1f(p.radius, radius);
                gl.useProgram(null);
            }
        }
        );

    let depthBuffer = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, depthBuffer);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, size, size);
    gl.bindRenderbuffer(gl.RENDERBUFFER, null);

/* BEGIN SIMULATION */
    function step() {
        empty(vertices);
        empty(indices);
        for (var i = 0; i < water.length; i++){
            vertices.push(water[i].newposition[0]);
            vertices.push(water[i].newposition[1]);
            vertices.push(water[i].newposition[2]);
            indices.push(i);
        }
        pts = createPoints(gl, vertices, indices);
        const loopCounter = 6;//4
        //The first for loop apply force and predict position
        for(var round = 0; round < 2; round++){
            for (let index = 0; index < water.length; index++) {
                omega(index, water, h);
                var c_force = corr_force(index, water, h, rho0, e1);

                water[index].velocity[0] += deltat*(GRAVITY[0] + c_force[0]);
                water[index].velocity[1] += deltat*(GRAVITY[1] + c_force[1]);
                water[index].velocity[2] += deltat*(GRAVITY[2] + c_force[2]);

                water[index].newposition[0] += water[index].velocity[0]*deltat;
                water[index].newposition[1] += water[index].velocity[1]*deltat;
                water[index].newposition[2] += water[index].velocity[2]*deltat;

            }
            //The second for loop, all the particles have new positions, fit them again into the grids, and then find their neighbors
            set_grid(water, grid);
            find_neighbor(water, grid, condition);

            //The third while loop
            var countLoop = 0;
            while (countLoop < loopCounter) {
                for (let index = 0; index < water.length; index++) {
                    let d_p = delta_p(index, water, h, e, k, n, delta_q, rho0)
                    if ((water[index].newposition[0] + d_p[0]) >= (length / gain)) {
                        water[index].newposition[0] = length / gain - water[index].newposition[2] / 10000;
                        water[index].velocity[0] = 0;
                    } else if ((water[index].newposition[0] + d_p[0]) <= 0) {
                        water[index].newposition[0] = 0 + water[index].newposition[2] / 10000;
                        water[index].velocity[0] = 0;
                    } else {
                        water[index].newposition[0] += d_p[0];
                        water[index].velocity[0] = water[index].velocity[0];
                    }
                    if ((water[index].newposition[1] + d_p[1]) >= (width / gain)) {
                        water[index].newposition[1] = width / gain - water[index].newposition[2] / 10000;
                        water[index].velocity[1] = 0;
                    } else if ((water[index].newposition[1] + d_p[1]) <= 0) {
                        water[index].newposition[1] = 0 + water[index].newposition[2] / 10000;
                        water[index].velocity[1] = 0;
                    } else {
                        water[index].newposition[1] += d_p[1];
                        water[index].velocity[1] = water[index].velocity[1];
                    }
                    if ((water[index].newposition[2] + d_p[2]) >= (height / gain)) {
                        water[index].newposition[2] = height / gain;
                        water[index].velocity[2] = 0;
                    } else if ((water[index].newposition[2] + d_p[2]) <= 0) {
                        water[index].newposition[2] = 0;
                        water[index].velocity[2] = 0;
                    } else {
                        water[index].newposition[2] += d_p[2];
                        water[index].velocity[2] = water[index].velocity[2];
                    }
                }
                countLoop++;
            }

            for (let index = 0; index < water.length; index++) {
                //const MAX_V = 3.2;
                var temp_v = cor_v(index, water, c, h);
                water[index].velocity[0] = (water[index].newposition[0]-water[index].oldposition[0])/deltat;
                water[index].velocity[1] = (water[index].newposition[1]-water[index].oldposition[1])/deltat;
                water[index].velocity[2] = (water[index].newposition[2]-water[index].oldposition[2])/deltat;
                water[index].velocity[0] += temp_v[0];
                water[index].velocity[1] += temp_v[1];
                water[index].velocity[2] += temp_v[2];
                // if(Math.abs(water[index].velocity[0])>MAX_V){
                //   water[index].velocity[0] = water[index].velocity[0] / Math.abs(water[index].velocity[0]) * MAX_V;
                // }
                // if(Math.abs(water[index].velocity[1])>MAX_V){
                //   water[index].velocity[1] = water[index].velocity[1] / Math.abs(water[index].velocity[1]) * MAX_V;
                // }
                // if(Math.abs(water[index].velocity[2])>MAX_V){
                //   water[index].velocity[2] = water[index].velocity[2] / Math.abs(water[index].velocity[2]) * MAX_V;
                // }
                water[index].oldposition[0] = water[index].newposition[0];
                water[index].oldposition[1] = water[index].newposition[1];
                water[index].oldposition[2] = water[index].newposition[2];
            }
        }
    }
    step();
/* END SIMULATION */

/* BEGIN RENDERING FUNCTIONS */
    function renderThickness() {
        thickBuffer.write(() => {
            gl.useProgram(thicknessProg);
            gl.clearColor(0.0, 0.0, 0.0, 1.0);
            gl.clear(gl.COLOR_BUFFER_BIT);

            gl.enable(gl.BLEND);
            gl.blendFunc(gl.ONE, gl.ONE);
            drawPoints(gl, thicknessProg, pts);
            gl.disable(gl.BLEND);
            gl.flush();

            gl.useProgram(null);
        });
    }

    function smoothThickness() {
        thickBuffer.write(() => {
            gl.useProgram(gaussianProg);
            gl.uniform2f(gaussianProg.blurDir, 1.0, 0.0);
            drawQuad(gaussianProg, thickBuffer.read());
            gl.flush();
        });

        thickBuffer.write(() => {
            gl.useProgram(gaussianProg);
            gl.uniform2f(gaussianProg.blurDir, 0.0, 1.0);
            drawQuad(gaussianProg, thickBuffer.read());
            gl.flush();
        });
    }

    function renderPositions() {
        img.write(() => {
            gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT,
                gl.RENDERBUFFER, depthBuffer);
            gl.useProgram(depthProg);

            gl.clearColor(1.0, 0.0, 1.0, 0.0); // color key requires +b
            gl.clearDepth(1.0);
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

            gl.enable(gl.DEPTH_TEST);
            drawPoints(gl, depthProg, pts);
            gl.disable(gl.DEPTH_TEST);
            gl.flush();

            gl.useProgram(null);
            gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT,
                gl.RENDERBUFFER, null);
        });
    }

    function smoothPositions() {
        img.write(() => {
            gl.useProgram(smoothProg);
            gl.uniform2f(smoothProg.blurDir, 1.0, 0.0);
            drawQuad(smoothProg, img.read());
            gl.flush();
        });

        img.write(() => {
            gl.useProgram(smoothProg);
            gl.uniform2f(smoothProg.blurDir, 0.0, 1.0);
            drawQuad(smoothProg, img.read());
            gl.flush();
        });
    }

    function shadePositions() {
        img.write(() => {
            gl.useProgram(normalProg);
            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_2D, thickBuffer.read());
            gl.activeTexture(gl.TEXTURE2);
            gl.bindTexture(gl.TEXTURE_CUBE_MAP, skyboxTexture);
            drawQuad(normalProg, img.read());
            gl.flush();
        });
    }

    function drawImage() {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        drawQuad(copyProg, img.read());
        gl.disable(gl.BLEND);
    }

    function renderer() {
        const selector = $("#renderMode");
        let fn;
        selector.change(e => {
            switch (e.target.value) {
            case "THK_SPHERES":
                fn = () => {
                    renderThickness();
                    img.write(() => drawQuad(copyProg, thickBuffer.read()));
                }
                break;
            case "THK_SMOOTH":
                fn = () => {
                    renderThickness();
                    smoothThickness();
                    img.write(() => drawQuad(copyProg, thickBuffer.read()));
                }
                break;
            case "POS_SPHERES":
                fn = renderPositions;
                break;
            case "POS_SMOOTH":
                fn = () => {
                    renderPositions();
                    smoothPositions();
                }
                break;
            case "POS_NORMALS":
                gl.useProgram(normalProg);
                gl.uniform1i(normalProg.normalOnly, 1);
                gl.useProgram(null);
                fn = () => {
                    renderPositions();
                    smoothPositions();
                    shadePositions();
                }
                break;
            case "FULL":
                gl.useProgram(normalProg);
                gl.uniform1i(normalProg.normalOnly, 0);
                gl.useProgram(null);
                fn = () => {
                    renderPositions();
                    smoothPositions();
                    renderThickness();
                    smoothThickness();
                    shadePositions();
                }
                break;
            }
        });
        selector.val("FULL");
        selector.change();
        return () => fn();
    }
    let render = renderer();
/* END RENDERING FUNCTIONS */

    const envCheckbox = $("#environment");
    function updateWebGL() {
        step();
        render();
        if (envCheckbox.prop("checked"))
            drawSkybox();
        else {
            gl.clearColor(1.0, 0.9, 0.6, 1.0);
            gl.clear(gl.COLOR_BUFFER_BIT);
        }
        drawImage();
        window.requestAnimationFrame(updateWebGL);
    }
    window.requestAnimationFrame(updateWebGL);
}

function main() {
    let queue = new createjs.LoadQueue();
    queue.on("complete", () => runCanvas(queue), this);
    let cubemapdir = "./skybox";
    let manifest = [];
    for (const s of ["x", "y", "z"]) {
        manifest.push({ id: "POSITIVE_" + s.toUpperCase()
                      , src: cubemapdir + "/pos" + s + ".jpg" });
        manifest.push({ id: "NEGATIVE_" + s.toUpperCase()
                      , src: cubemapdir + "/neg" + s + ".jpg" });
    }
    queue.loadManifest(manifest);
}

main();
</script>

</body>
</html>

<!-- vim: set ts=4 sts=4 sw=4 et: -->

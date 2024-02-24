#include "framework.h"

float epsilon = 0.0001f;

vec3 operator/(vec3 left, vec3 right) {
	return vec3(left.x / right.x, left.y / right.y, left.z / right.z);
}

enum MaterialType{ROUGH, REFLECTIVE};
struct Material {
	vec3 kd, ka, ks;
	float shine;
	vec3 F0;
	MaterialType type;
	Material(MaterialType mt) : type(mt) {}
};
struct RoughMaterial : public Material {
	RoughMaterial(vec3 kd_, vec3 ks_, float shine_) : Material(ROUGH) {
		kd = kd_;
		ka = kd_ * M_PI;
		ks = ks_;
		shine = shine_;
	}
};
struct ReflectiveMaterial : public Material {
	ReflectiveMaterial(vec3 n, vec3 k) : Material(REFLECTIVE) {
		vec3 I(1, 1, 1);
		F0 = ((n - I) * (n - I) + k * k) / ((n + I) * (n + I) + k * k);
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material = nullptr;
	Hit() { t = -1; }
};
struct Ray {
	vec3 start, dir;
	Ray(vec3 start_, vec3 dir_) : start(start_), dir(normalize(dir_)) {}
};

class Intersectable {
protected:
	Material* m;
public:
	std::string name;
	virtual Hit intersect(const Ray&) = 0;
};

class QuadraticS : public Intersectable {
protected:
	mat4 Q;

public:
	QuadraticS() {}
	QuadraticS(Material* m_) {
		m = m_;
	}

	float f(vec4 r) {
		return dot(r * Q, r);
	}

	std::vector<float> rootsOfIntersect(const Ray& r) {
		vec4 S{ r.start.x, r.start.y, r.start.z, 1 };
		vec4 D{ r.dir.x, r.dir.y, r.dir.z, 0 };

		float a = dot((D * Q), D);
		float b = dot((D * Q), S) + dot((S * Q), D);
		float c = dot((S * Q), S);
		float disc = b * b - (4.0f * a * c);
		if (disc < 0)
			return std::vector<float> {-1.0f, -1.0f};
		// t1 >= t2
		float t1 = (-b + sqrt(disc)) / (2.0f * a);
		float t2 = (-b - sqrt(disc)) / (2.0f * a);

		return std::vector<float> {t1, t2};
	}

	float getClosestValidP(std::vector<float> points) {
		float extremeV = -1.0;
		// maximum search for correct operation of the algorithm
		for (auto p : points)
			if (p > extremeV)
				extremeV = p;
		// find the smallest positive number in the array
		for (auto p : points)
			if (p > 0 && p < extremeV)
				extremeV = p;
		return extremeV;
	}

	virtual std::vector<float> limitTop(const Ray& r, std::vector<float> roots) { return roots; };
	virtual std::vector<float> limitRoom(const Ray& r, std::vector<float> roots) { return roots; };
	virtual std::vector<float> coverTop(const Ray& r, std::vector<float> roots) { return roots; };

	Hit intersect(const Ray& r) {
		Hit hit;
		std::vector<float> roots = rootsOfIntersect(r);
		roots = limitRoom(r, roots);
		roots = limitTop(r, roots);
		roots = coverTop(r, roots);
		hit.t = getClosestValidP(roots);

		hit.position = r.start + r.dir * hit.t;
		vec4 homPos(hit.position.x, hit.position.y, hit.position.z, 1);
		hit.normal = normalize(vec3((homPos * Q * 2.0f).x, (homPos * Q * 2.0f).y, (homPos * Q * 2.0f).z));
		hit.material = m;
		return hit;
	}
};

class Room : public QuadraticS {
	vec4 roof;
	float a, b, c, pZ0;
public:
	Room( Material* m_) {
		name = "szoba";
		m = m_;
		a = 1.7f, b = 2.0f, c = 1.0f;
		Q =					{ b*b*c*c, 0.0f, 0.0f, 0.0f,
							  0.0f, a*a*c*c, 0.0f, 0.0f,
							  0.0f, 0.0f, a*a*b*b,	0.0f,
							  0.0f, 0.0f, 0.0f, -a*a*b*b*c*c };
		pZ0 = -0.95f;
		roof = vec4(0.0f, 0.0f, 1.0f, pZ0);
	}
	mat4 getQ() { return Q; }

	virtual std::vector<float> limitTop(const Ray& r, std::vector<float> roots) {
		for (int i = 0; i < roots.size(); i++) {
			vec3 hit(r.start + r.dir * roots[i]);
			vec4 p(hit.x, hit.y, hit.z, 1);
			if (dot(p, roof) > 0)
				roots[i] = -1;
		}

		return roots;
	}

	float A() { return a; }
	float B() { return b; }
	float Z0() { return pZ0; }
};
class Paraboloid : public QuadraticS {
	mat4 qRoom;
public:

	Paraboloid(Material *m_, mat4 qRoom_) {
		name = "parabola";
		m = m_;
		qRoom = qRoom_;
		float x0, y0, a, c;
		x0 = -0.7f;
		y0 = 0.1f;
		a = 0.15f;
		c = 0.5f;
		Q = { c, 0.0f, 0.0f, -c * x0,
			0.0f, c, 0.0f, -c * y0,
			0.0f, 0.0f, 0.0f,	0.5f * a,
			-c * x0, -c * y0, 0.5f * a, c * (x0 * x0 + y0 * y0) };
	}

	virtual std::vector<float> limitRoom(const Ray& r, std::vector<float> roots) {
		for (int i = 0; i < roots.size(); i++) {
			vec3 hit(r.start + r.dir * roots[i]);
			vec4 p(hit.x, hit.y, hit.z, 1);
			if (dot(p * qRoom, p) > 0)
				roots[i] = -1;
		}

		return roots;
	}

};
class Cylinder : public QuadraticS {
	mat4 qRoom;
	vec4 limitP;
public:

	Cylinder(Material* m_, mat4 qRoom_) {
		name = "cilinder";
		m = m_;
		qRoom = qRoom_;
		float x0 = 0.9f, y0 = -0.1f, r = 0.30f;
		Q = { 1.0f, 0.0f, 0.0f, -x0,
			  0.0f, 1.0f, 0.0f, -y0,
			  0.0f, 0.0f, 0.0f, 0.0f,
			 -x0, -y0, 0.0f, x0*x0 + y0*y0 - r*r};

		float pZ0 = +0.25f;
		limitP = vec4(0.0f, 0.0f, 1.0f, pZ0);
	}

	virtual std::vector<float> limitTop(const Ray& r, std::vector<float> roots) {
		for (int i = 0; i < roots.size(); i++) {
			vec3 hit(r.start + r.dir * roots[i]);
			vec4 p(hit.x, hit.y, hit.z, 1);
			if (dot(p, limitP) > epsilon)
				roots[i] = -1;
		}

		return roots;
	}
	virtual std::vector<float> limitRoom(const Ray& r, std::vector<float> roots) {
		for (int i = 0; i < roots.size(); i++) {
			vec3 hit(r.start + r.dir * roots[i]);
			vec4 p(hit.x, hit.y, hit.z, 1);
			if (dot(p * qRoom, p) > 0)
				roots[i] = -1;
		}

		return roots;
	}
	virtual std::vector<float> coverTop(const Ray& r, std::vector<float> roots) {
	// check if the normal vector and the ray vector are parallel
		vec4 S{ r.start.x, r.start.y, r.start.z, 1 };
		vec4 D{ r.dir.x, r.dir.y, r.dir.z, 0 };
		float d = dot(D, limitP);
		float root;
		if (abs(d) > epsilon) {
			root = (dot(S, limitP)) / -d ;
			vec3 possibleHit(r.start + r.dir * root);
			vec4 p(possibleHit.x, possibleHit.y, possibleHit.z, 1);
			if (f(p) < epsilon && root > 0) {
				roots.push_back(root);
			}
		}
		return roots;
	}

	Hit intersect(const Ray& r) {
		Hit hit;
		std::vector<float> roots = rootsOfIntersect(r);
		roots = limitRoom(r, roots);
		roots = limitTop(r, roots);
		roots = coverTop(r, roots);
		float pSide = getClosestValidP(roots);
		float pTop = coverTop(r, roots)[0];
		if (abs(pTop -  pSide) < epsilon) {
			hit.t = pTop;
			hit.normal = vec3(0.0f, 0.0f, 1.0f);
			hit.position = r.start + r.dir * hit.t;
		}
		else {
			hit.t = pSide;
			hit.position = r.start + r.dir * hit.t;
			vec4 homPos(hit.position.x, hit.position.y, hit.position.z, 1);
			hit.normal = normalize(vec3((homPos * Q * 2.0f).x, (homPos * Q * 2.0f).y, (homPos * Q * 2.0f).z));
		}

		hit.material = m;
		return hit;
	}
};
class HiperBoloid : public QuadraticS {
	mat4 qRoom;
	vec4 limitP;
public:

	HiperBoloid(Material* m_, mat4 qRoom_) {
		name = "hiperbola";
		m = m_;
		qRoom = qRoom_;
		float x0 = 0.35f, y0 = 0.75f, z0 = 0.2f, a = 0.16f, b = 0.16f, c = 0.4f;
		Q = { b*b*c*c, 0.0f, 0.0f, -x0*b*b*c*c,
			0.0f, a*a*c*c, 0.0f, -y0*a*a*c*c,
			0.0f, 0.0f, -a*a*b*b, z0*a*a*b*b,
			-x0 *b*b*c*c, -y0*a*a*c*c, z0*a*a*b*b, x0*x0*b*b*c*c+y0*y0*a*a*c*c-z0*z0*a*a*b*b - a*a*b*b*c*c };

		float pZ0 = -0.6f;
		limitP = vec4(0.0f, 0.0f, 1.0f, pZ0);
	}

	virtual std::vector<float> limitTop(const Ray& r, std::vector<float> roots) {
		for (int i = 0; i < roots.size(); i++) {
			vec3 hit(r.start + r.dir * roots[i]);
			vec4 p(hit.x, hit.y, hit.z, 1);
			if (dot(p, limitP) > 0)
				roots[i] = -1;
		}

		return roots;
	}
	virtual std::vector<float> limitRoom(const Ray& r, std::vector<float> roots) {
		for (int i = 0; i < roots.size(); i++) {
			vec3 hit(r.start + r.dir * roots[i]);
			vec4 p(hit.x, hit.y, hit.z, 1);
			if (dot(p * qRoom, p) > 0)
				roots[i] = -1;
		}

		return roots;
	}
	virtual std::vector<float> coverTop(const Ray& r, std::vector<float> roots) {
		// check if the normal vector and the ray vector are parallel
		vec4 S{ r.start.x, r.start.y, r.start.z, 1 };
		vec4 D{ r.dir.x, r.dir.y, r.dir.z, 0 };
		float d = dot(D, limitP);
		float root;
		if (fabs(d) > epsilon) {
			root = (dot(S, limitP)) / -d;
			vec3 possibleHit(r.start + r.dir * root);
			vec4 p(possibleHit.x, possibleHit.y, possibleHit.z, 1);
			if (f(p) < 0) {
				roots.push_back(root);
			}
		}
		return roots;
	}
};

class SolarPipe : public QuadraticS {
	mat4 qRoom;
	vec4 limitP, limitP2;
public:

	SolarPipe(Material* m_, mat4 qRoom_, float a_, float b_, float z0_) {
		name = "napfenycso";
		m = m_;
		qRoom = qRoom_;
		float x0 = 0.0f, y0 = 0.0f, z0 = 1.0f, a = a_ * sqrtf(0.19f), b = b_ * sqrtf(0.19f), c = 1.0f;
		Q = { b * b * c * c, 0.0f, 0.0f, -x0 * b * b * c * c,
			0.0f, a * a * c * c, 0.0f, -y0 * a * a * c * c,
			0.0f, 0.0f, -a * a * b * b, z0 * a * a * b * b,
			-x0 * b * b * c * c, -y0 * a * a * c * c, z0 * a * a * b * b, x0 * x0 * b * b * c * c + y0 * y0 * a * a * c * c - z0 * z0 * a * a * b * b - a * a * b * b * c * c };

		float pZ0 = -1.8f;
		limitP = vec4(0.0f, 0.0f, 1.0f, pZ0);
		limitP2 = vec4(0.0f, 0.0f, -1.0f, -z0_);
	}

	virtual std::vector<float> limitTop(const Ray& r, std::vector<float> roots) {
		for (int i = 0; i < roots.size(); i++) {
			vec3 hit(r.start + r.dir * roots[i]);
			vec4 p(hit.x, hit.y, hit.z, 1);
			if (dot(p, limitP) > 0 || dot(p, limitP2) > 0)
				roots[i] = -1;
		}

		return roots;
	}

	Hit intersect(const Ray& r) {
		Hit hit;
		std::vector<float> roots = rootsOfIntersect(r);
		roots = limitRoom(r, roots);
		roots = limitTop(r, roots);
		roots = coverTop(r, roots);
		hit.t = getClosestValidP(roots);

		hit.position = r.start + r.dir * hit.t;
		vec4 homPos(hit.position.x, hit.position.y, hit.position.z, 1);
		hit.normal = -1.0f * normalize(vec3((homPos * Q * 2.0f).x, (homPos * Q * 2.0f).y, (homPos * Q * 2.0f).z));
		hit.material = m;
		return hit;
	}
};

class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 eye_, vec3 lookat_, vec3 vup, float fov_) {
		eye = eye_;
		lookat = lookat_;
		fov = fov_;

		vec3 w = eye - lookat;
		float windowSize = length(w) * tanf(fov / 2); // 1:1 screen ratio
		right = normalize(cross(vup, w)) * windowSize;
		up = normalize(cross(w, right)) * windowSize;
	}

	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1);
		return Ray(eye, dir);
	}
};

struct Light {
	vec3 dir, Le;
	Light() {}
	Light(vec3 dir_, vec3 Le_) : dir{ normalize(dir_) }, Le{ Le_ } {}
};

class Scene {
	std::vector<Intersectable*> objs;
	std::vector<vec3> sampleLights;
	Light sun;
	Camera cam;
	vec3 La;

	float surfaceEllipse;
	int sampleNum;
	vec3 normalEllipse;
public:
	bool inEllipse(float x, float y, float a, float b) {
		float f = ((x * x) / (a * a)) + ((y * y) / (b * b)) - 1.0f;
		return (f < epsilon);
	}

	void build() {

		/// kamera be�ll�t�sok
		vec3 eye = vec3(0.0f, -1.65f, 0.4f);	// camera position
		vec3 vup = vec3(0, 0, 1);				// UP direction
		vec3 lookat = vec3(0, 2.0f, -0.4f);		// we are looking at this point
		float fov = 70 * M_PI / 180;			// field of view
		cam.set(eye, lookat, vup, fov);

		La = vec3(0.2f, 0.2f, 0.2f);
		sun = Light(vec3(-5, 0, 2), vec3(2, 2, 2));

		/// materials (1 gold + 1 silver + 2 diffuse + room)
		ReflectiveMaterial* silver = new ReflectiveMaterial(vec3(0.14f, 0.16f, 0.13f), vec3(4.1f, 2.3f, 3.1f));
		ReflectiveMaterial* gold = new ReflectiveMaterial(vec3(0.17f, 0.35f, 1.5f), vec3(3.1f, 2.7f, 1.9f));
		RoughMaterial* mRoom = new RoughMaterial(vec3(0.545f, 0.0f, 0.545f), vec3(2, 2, 2), 70.0f);
		RoughMaterial* mCylinder = new RoughMaterial(vec3(0.235, 0.702, 0.443), vec3(1, 1, 1), 50.0f);
		RoughMaterial* mHiperboloid = new RoughMaterial(vec3(0.633, 0.510, 0.0f), vec3(2, 2, 2), 70.0f);

		/// initialize objects
		Room* room = new Room(mRoom);
		objs.push_back(room);
		objs.push_back(new Paraboloid(gold, room->getQ()));
		objs.push_back(new Cylinder(mCylinder, room->getQ()));
		objs.push_back(new HiperBoloid(mHiperboloid, room->getQ()));
		objs.push_back(new SolarPipe(silver, room->getQ(), room->A(), room->B(), room->Z0()));

		sampleNum = 20;
		int generated = 0;
		srand(0);
		float scalingFactor = sqrtf(1.0f - (room->Z0() * room->Z0()));
		float a = room->A() * scalingFactor;
		float b = room->B() * scalingFactor;
		surfaceEllipse = a * b * M_PI;
		normalEllipse = normalize(vec3(0, 0, 1));
		while (generated < sampleNum) {
			float px = (float)rand() / (RAND_MAX / (2.0f * a));
			float py = (float)rand() / (RAND_MAX / (2.0f * b));
			float z = room->Z0();
			if (inEllipse(px-a, py-b, a, b)) {
				sampleLights.push_back(vec3(px-a, py-a, -z));
				generated++;
			}
		}
	}

	void render(std::vector<vec4>& image) {
		for (int y = 0; y < windowHeight; y++)
#pragma omp paralell for
			for (int x = 0; x < windowWidth; x++) {
				vec3 color = trace(cam.getRay(x, y));
				image[y * windowWidth + x] = vec4(color.x, color.y, color.z, 1.0f);
			}
	}

	Hit firstIntersect(Ray ray) {
		Hit best;
		for (auto obj : objs) {
			Hit hit = obj->intersect(ray);
			if (hit.t > 0 && (best.t < 0 || hit.t < best.t))
				best = hit;
		}

		if (dot(ray.dir, best.normal) > 0)
			best.normal = best.normal  * -1;

		return best;
	}

	bool shadowIntersect(Ray ray) {
		for (auto obj : objs) {
			Hit h = obj->intersect(ray);
			if (h.t > 0)
				return true;
		}
		return false;
	}

	vec3 outerLight(const Ray& ray) {
		return vec3(4.0f, 4.0f, 4.8f) + vec3(7.2f, 7.2f, 7.2f) * powf(dot(ray.dir, sun.dir), 10);
	}

	float omega(const Ray& ray, vec3 target) {
		return surfaceEllipse / (float)((float) sampleNum) * dot(ray.dir, normalEllipse) / (length(ray.start - target) * length(ray.start - target));
	}

	vec3 trace(const Ray& ray, int depth = 0) {
		if (depth > 5)
			La;

		Hit hit = firstIntersect(ray);
		if (hit.t < 0)
			return outerLight(ray);

		vec3 outrad(0, 0, 0);

		if (hit.material->type == ROUGH) {
			outrad = hit.material->ka * La;
			// for all light source:
			for (auto l : sampleLights) {
				vec3 lightDir(normalize(l - hit.position /*+ hit.normal * epsilon*/));
				Ray diffuseRay(hit.position + hit.normal * epsilon, lightDir);
				float cosTheta = dot(hit.normal, diffuseRay.dir);
				if (cosTheta > 0 && !shadowIntersect(diffuseRay)) {
					vec3 traced = trace(diffuseRay, depth+1);
					outrad = outrad + traced * hit.material->kd * cosTheta * omega(diffuseRay, l);
					vec3 halfway = normalize(-ray.dir + diffuseRay.dir);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0)
						outrad = outrad + traced * hit.material->ks * powf(cosDelta, hit.material->shine) * omega(diffuseRay, l);
				}
			}
		}

		if (hit.material->type == REFLECTIVE) {
			vec3 reflectionDir = ray.dir - 2.0f * hit.normal * dot(hit.normal, ray.dir);
			vec3 fresnel = hit.material->F0 + (vec3(1, 1, 1) - hit.material->F0) * powf(1 + dot(ray.dir, hit.normal), 5);
			Ray reflectRay(hit.position + hit.normal * epsilon, reflectionDir);
			outrad = outrad + trace(reflectRay, depth + 1) * fresnel;
		}

		return outrad;
	}

};

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char* const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	layout(location = 0) in vec2 cVP;	// Varying input: vp = vertex position is expected in attrib array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVP + vec2(1,1))/2;
		gl_Position = vec4(cVP.x, cVP.y, 0, 1);
	}
)";

// fragment shader in GLSL
const char* const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers

	uniform sampler2D textureUnit;		// uniform variable, the color of the primitive
	in vec2 texcoord;
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = texture(textureUnit, texcoord);	// computed color is the color of the primitive
	}
)";

Scene scene;
GPUProgram gpuProgram;

class FullscreenTexturedQuad {
	unsigned int vao = 0, textureID = 0;
public:
	FullscreenTexturedQuad() {
		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer object
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		float vertices[] = { -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		glGenTextures(1, &textureID);
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	void Load(std::vector<vec4>& image){
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]);
	}

	void Draw() {
		glBindVertexArray(vao);		// make it active
		int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
		const unsigned int textureUnit = 0;
		if (location >= 0) {
			glUniform1i(location, textureUnit);
			glActiveTexture(GL_TEXTURE0 + textureUnit);
			glBindTexture(GL_TEXTURE_2D, textureID);
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

FullscreenTexturedQuad* ftq;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	ftq = new FullscreenTexturedQuad();
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	std::vector<vec4> image(windowWidth*windowHeight);
	scene.render(image);
	ftq->Load(image);
	ftq->Draw();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {}
void onKeyboardUp(unsigned char key, int pX, int pY) {}
void onMouseMotion(int pX, int pY) {}
void onMouse(int x, int y, int cx, int cy) {}
void onIdle() {}

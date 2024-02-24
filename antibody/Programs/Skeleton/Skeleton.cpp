#include "framework.h"

template <class T> struct Dnum {
	float f;
	T d;
	Dnum(float f0 = 0, T d0 = T(0)): f(f0), d(d0) {}
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) { return Dnum(f * r.f, f * r.d + d * r.f); }
	Dnum operator/(Dnum r) { return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f); }

};

template <class T> Dnum<T> Sin(Dnum<T> g) { return Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template <class T> Dnum<T> Cos(Dnum<T> g) { return Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template <class T> Dnum<T> Sinh(Dnum<T> g) { return Dnum<T>(sinhf(g.f), coshf(g.f) * g.d); }
template <class T> Dnum<T> Cosh(Dnum<T> g) { return Dnum<T>(coshf(g.f), sinhf(g.f) * g.d); }
template <class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }

typedef Dnum<vec2> Dnum2;

vec4 normalize(vec4 v) {
	return v * (1 / sqrtf(dot(v, v)));
}



class Camera {
public:
	float fov, asp, fp, bp;	   // intrinsic parameters
	vec3  wEye, wLookat, wVup; // extrinsic parameters
	Camera() {
		asp = (float) windowWidth/windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 1.0f, bp = 20.0f;
	}
	mat4 V() { // view matrix
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) *
			mat4(u.x, v.x, w.x, 0,
				 u.y, v.y, w.y, 0,
				 u.z, v.z, w.z, 0,
				 0,   0,   0,   1);
	}
	mat4 P() { // projection matrix
		float sy = 1 / tan(fov / 2);
		return mat4(sy / asp, 0,  0,                       0,
			        0,        sy, 0,                       0,
			        0,        0,   -(fp + bp) / (bp - fp),-1,
			        0,        0, -2 * fp * bp / (bp - fp), 0);
	}
};

struct Material {
	vec3 kd, ks, ka;
	float shine;
};

struct Light {
	vec3 La, Le;			// ambient light or point-wise / directional
	vec4 wLightPos;		// position of the light source (if w = 0, directional else point-wise)
};

class StripedTexture : public Texture {
public:
	StripedTexture(const int w, const int h) : Texture() {
		std::vector<vec4> image(w * h);
		const vec4 paleyellow(0.94f, 0.90f, 0.55f, 1.0f), deeppurple(0.29f, 0.0f, 0.51f, 1.0f);
		for(int x = 0; x < w; x++)
			for (int y = 0; y < h; y++) {
				image[y * w + x] = (x % 2 < 1) ? paleyellow : deeppurple;
			}
		create(w, h, image, GL_NEAREST);
	}
};

class CheckedTexture : public Texture {
public:
	CheckedTexture(const int w, const int h) : Texture() {
		std::vector<vec4> image(w * h);
		const vec4 steelblue(0.28f, 0.51f, 0.70f, 1.0f), brickred(0.86f, 0.08f, 0.23f, 1.0f);
		for (int x = 0; x < w; x++)
			for (int y = 0; y < h; y++) {
				image[y * w + x] = ((x & 1) ^ (y & 1)) ? steelblue : brickred;
			}
		create(w, h, image, GL_NEAREST);
	}
};

struct RenderState {
	mat4				MVP, M, Minv, V, P;
	Material*			material;
	std::vector<Light>	lights;
	Texture*			texture;
	vec3				wEye;
};

class Shader : public GPUProgram {
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shine, name + ".shine");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".la");
		setUniform(light.Le, name + ".le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};

class PhongShader : public Shader {
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 la, le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;
		uniform int nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3 vtxPos;  // pos in model sp
		layout(location = 1) in vec3 vtxNorm; // normal in mod sp
		layout(location = 2) in vec2 vtxUV;

		out vec3 wNormal;           // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];            // light dir in world space
		out vec2 texcoord;

		void main() {
		   gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
		   vec4 wPos = vec4(vtxPos, 1) * M;
		   for(int i = 0; i < nLights; i++){
				wLight[i]  = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
		   }
		   wView   = wEye * wPos.w - wPos.xyz;
		   wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		   texcoord = vtxUV;
		}

		)";
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 la, le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shine;
		};

		uniform Material material;
		uniform Light[8] lights;
		uniform int nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dirs
		in  vec2 texcoord;

		out vec4 fragmentColor; // output goes to frame buffer

		void main() {
		   vec3 N = normalize(wNormal);
		   vec3 V = normalize(wView);
		   if(dot(N, V) < 0) N = -N;

  		   vec3 texColor = texture(diffuseTexture, texcoord).rgb;
		   vec3 ka = material.ka * texColor;
		   vec3 kd = material.kd * texColor;

		   vec3 radiance = vec3(0,0,0);
		   for(int i = 0; i < nLights; i++) {
			   vec3 L = normalize(wLight[i]);
			   vec3 H = normalize(L + V);
			   float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
		       radiance += ka * lights[i].la + (kd * texColor * cost + material.ks * pow(cosd, material.shine)) * lights[i].le;
		   }
		   fragmentColor = vec4(radiance, 1);
		}
		)";
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use();
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

struct VertexData {
	vec3 position, normal;
	vec2 texcoord;
	VertexData() {}
	VertexData(vec3 p, vec3 n, vec2 tc) {
		position = p;
		normal = n;
		texcoord = tc;
	}
};

class Geometry {
protected:
	unsigned int vao, vbo;
	float time = 0;
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() = 0;
	virtual void reCreate() = 0;
	virtual float radOfCircumSphere() = 0;
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
	void updateTime(float t) {
		time = (t < 100 * M_PI)? t : 0;
	}
	float getTime() { return time; }
};

// regular tetrahedron
class recTetraeder : public Geometry {
	struct Side {
		std::vector<vec3> vtxs;
		vec3 normal;
		Side(vec3 a, vec3 b, vec3 c, vec3 n) {
			vtxs.push_back(a);
			vtxs.push_back(b);
			vtxs.push_back(c);
			normal = n;
		}
	};

	std::vector<Side> sides;
	int nVtx = 9;
	std::vector<VertexData> vtxData;
	std::vector<recTetraeder*> children;

	std::vector<vec3> currentbase;
	float currentheight;
	vec3 shiftCenter;
	float R;

public:
	virtual void reCreate() {
		freeChildren();
		for (auto s : sides)
			children.push_back(new recTetraeder(s.vtxs, sqrtf(2.0f) * 0.7f +  0.7f * -cosf(3.0f * time), 2));
	}

	recTetraeder() {
		float height = sqrtf(2.0f);
		// an alpha angle is enclosed by the segment pointing to one of the vertices and one of the faces of the tetrahedron from the base point of the triangle
		// I recorded the mentioned section for exactly one leg length
		// so tg(alpha) = h / 1 --> alpha = atanf(sqrtf(2.0f)) this is true for all regular tetrahedrons -> with h = sqrtf(2) we get a regular tetrahedron

		// further in regular tetrahedron we get the base point of the tetrahedron S= a+b+c+d / 4 (vertices of abcd)
		// radius of circumcircle length(S - center)
		// here now center = (0,0,0), d = center + n * h, n = (0,0,1), h = sqrt(2) -> d = (0,0,sqrt( 2)

		std::vector<vec3> initbase;
		initbase.push_back(vec3(0, 1, 0));
		initbase.push_back(vec3(sqrtf(3)/2.0f, -0.5f, 0));
		initbase.push_back(vec3(-sqrtf(3)/2.0f, -0.5f, 0));

		// radius of circumcircle
		vec3 S((initbase[0] + initbase[0] + initbase[0] + vec3(0, 0, height)) / 4.0f);
		shiftCenter = S - vec3(0, 0, 0);
		R = length(shiftCenter);

		Create(initbase, true, height);

		for (auto s : sides)
			children.push_back(new recTetraeder(s.vtxs, height * 0.6f, 3));

	}

	recTetraeder(std::vector<vec3> base, float height, int maxdepth) {
		currentbase = base, currentheight = height;
		Create(base, false, height);

		if(maxdepth > 0)
			for (auto s : sides)
				children.push_back(new recTetraeder(s.vtxs, height * 0.6f, maxdepth-1));
	}
	void Create(std::vector<vec3> base, bool first, float h) {
		vec3 normal = normalize(cross(base[2] - base[0], base[1] - base[0]));
		vec3 center((base[0] + base[1] + base[2]) / 3.0f);
		vec3 vertex = center + normal * h;

		vec3 v0((base[0] + base[1]) / 2.0f);
		vec3 v1((base[1] + base[2]) / 2.0f);
		vec3 v2((base[2] + base[0]) / 2.0f);

		vec3 n01c(normalize(cross(vertex - v0, v1 - v0))) , n12c(normalize(cross(vertex - v1, v2 - v1))), n20c(normalize(cross(vertex - v2, v0 - v2)));

		sides.push_back(Side(v0, v1, vertex, n01c));
		sides.push_back(Side(v1, v2, vertex, n12c));
		sides.push_back(Side(v2, v0, vertex, n20c));
		if (first) {
			sides.push_back(Side(v0, v2, v1, -normal));
			nVtx += 3;
		}

		for (auto s : sides)
			for(auto p: s.vtxs)
				vtxData.push_back(VertexData(p, s.normal, vec2(1,1)));

		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, nVtx * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0); // AttArr 0 = POSITION
		glEnableVertexAttribArray(1); // AttArr 1 = NORMAL
		glEnableVertexAttribArray(2); // AttArr 2 = UV

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw() {
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glDrawArrays(GL_TRIANGLES, 0, nVtx);

		for (auto c : children)
			c->Draw();
	}

	void freeChildren() {
		for (int i = 0; i < children.size(); i++) {
			children[i]->freeChildren();
			delete children[i];
		}
		children.clear();
	}

	float radOfCircumSphere() {
		// see above
		return R;
	}

	vec3 correctCenter() {
		return -shiftCenter;
	}
};


 int testellationlvl = 40;

class ParamSurface : public Geometry {
	unsigned int nVtxPerStrip, nStrips;

public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }
	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	VertexData GenVertexData(float u, float v) {
		VertexData vtxData;
		vtxData.texcoord = vec2(u, v);
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		vtxData.normal = normalize(cross(drdU, drdV));
		return vtxData;
	}
	void reCreate() { Create(); }
	void Create(int N = testellationlvl, int M = testellationlvl) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData; // CPU-n
		for (int i = 0; i < N; i++)
			for (int j = 0; j <= M; j++) {
			vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
			vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
		}

		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0); // AttArr 0 = POSITION
		glEnableVertexAttribArray(1); // AttArr 1 = NORMAL
		glEnableVertexAttribArray(2); // AttArr 2 = UV

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,	sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,	sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE,	sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw() {
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		for (unsigned int i = 0; i < nStrips; i++)
			glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};

// kell egy sphere a szob�nak + egy a virusnak, a virust kulon le kene szarmaztatni ebbol
class Sphere : public ParamSurface {
	virtual Dnum2 R(float u, float v, float t) {
		return Dnum2(1,0);
	}

public:
	Sphere() { Create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		Dnum2 r = R(U.f, V.f, time);

		U = U * 2.0f * (float)M_PI;
		V = V * (float)M_PI;
		X = r * Cos(U) * Sin(V);
		Y = r * Sin(U) * Sin(V);
		Z = r * Cos(V);
	}

	float radOfCircumSphere() { return R(0, 0, 0).f;}
};

class VirusBase : public Sphere {
	virtual Dnum2 R(float u, float v, float t) {
		return  Dnum2(2.0f) + Dnum2(0.35f) * (Dnum2(0.3f) + Sin(Dnum2(4)*Dnum2(t)) * Sin(Dnum2(5.5f) * Dnum2(M_PI) * (Dnum2(u, vec2(1,0)) + Dnum2(2.0f) * Dnum2(v, vec2(0,1)))));
	}
public:
	VirusBase() { Create(); }
};

class Tractricoid : public ParamSurface {
public:
	Tractricoid() { Create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		float height = 4.0f;

		U = U * height;
		V = V * 2.0f * (float)M_PI;
		X = Cos(V) / Cosh(U);
		Y = Sin(V) / Cosh(U);
		Z = U - Tanh(U);
	}

	float radOfCircumSphere() {
		return 0; // nem fontos
	}
};

struct Object {
		Shader* shader;
		Material* material;
		Texture* texture;
		Geometry* geometry;
		vec3 scale, pos, rotAxis;
		float rotAngle;
		std::vector<Object*> children;

	public:
		Object(Shader* shader_, Material* material_, Texture* texture_, Geometry* geometry_)
			: scale(vec3(1, 1, 1)), pos(vec3(0, 0, 0)), rotAxis(0, 0, 1), rotAngle(0) {
			shader = shader_;
			material = material_;
			texture = texture_;
			geometry = geometry_;
		}

		virtual void SetModellingTransform(mat4& M, mat4& Minv) {
			M = ScaleMatrix(scale) * RotationMatrix(rotAngle, rotAxis) * TranslateMatrix(pos);
			Minv = TranslateMatrix(-pos) * RotationMatrix(-rotAngle, rotAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
		}

		virtual void Draw(RenderState state) {
			mat4 M, Minv;
			SetModellingTransform(M, Minv);
			state.M = M;
			state.Minv = Minv;
			state.MVP = state.M * state.V * state.P;
			state.material = material;
			state.texture = texture;
			shader->Bind(state); // uniform v�ltoz�k be�ll�t�sa
			geometry->Draw(); // h�romsz�gek v�gigmennek a pipeline-on
			for (Object* child : children) child->Draw(state);
		}
		virtual void Animate(float ts, float te) {
		}

	};

struct Spike : public Object {
		mat4 parentM, parentMinv;
		vec3 n;
		vec2 texcoord;
		Spike(Shader* shader_, Material* material_, Texture* texture_, Geometry* geometry_, vec3 normal, mat4 pM, mat4 pMinv) : Object(shader_, material_, texture_, geometry_) {
			n = normal;
			parentM = pM;
			parentMinv = pMinv;
		}

		void setParentMs(mat4 parentM_, mat4 parentMinv_) {
			parentM = parentM_;
			parentMinv = parentMinv_;
		}

		virtual void SetModellingTransform(mat4& M, mat4& Minv) {
			M = ScaleMatrix(scale) * Normal2Rotation() * TranslateMatrix(pos) * TranslateMatrix(-0.3f * n) * parentM;
			Minv = parentMinv * TranslateMatrix(+0.3f * n) * TranslateMatrix(-pos) * N2Rinv() * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
		}

		mat4 Normal2Rotation() {
			float denom = sqrtf(n.x * n.x + n.y * n.y);
			return mat4(
				n.y / denom, -n.x / denom, 0, 0,
				n.x * n.z / denom, n.y * n.z / denom, -denom, 0,
				n.x, n.y, n.z, 0,
				0, 0, 0, 1);
		}

		mat4 N2Rinv() {
			float denom = sqrtf(n.x * n.x + n.y * n.y);
			return mat4(
				n.y / denom, n.x * n.z / denom, n.x, 0,
				-n.x / denom, n.y * n.z / denom, n.y, 0,
				0, -denom, n.z, 0,
				0, 0, 0, 1);
		}

	};

struct Virus : public Object {
		vec3 rotAxis2;
		float rotAngle2;

		vec4 currPos;
		float R;

		bool move = true;

		void updateRotParams(float t) {
			vec4 q(sinf(t / 2), sinf(t / 3), sinf(t / 5), cosf(t));
			normalize(q);
			rotAngle2 = acosf(q.w) * 2.0f;
			float denom = sinf(rotAngle2 / 2);
			if (fabs(denom) < 0.0001f) {
				// if the rotation is 0/360 degrees, it doesn't matter which axis it rotates
				rotAxis2 = vec3(q.x, q.y, q.z);
			}
			else {
				rotAxis2 = vec3(q.x / denom, q.y / denom, q.z / denom);
			}
		}

		Virus(Shader* shader_, Material* material_, Texture* texture_, Geometry* geometry_) : Object(shader_, material_, texture_, geometry_) {
			updateRotParams(0);
			pos = vec3(2, 0, 2);
			scale = vec3(0.7, 0.7, 0.7);
			R = geometry->radOfCircumSphere() * ((scale.x + scale.y + scale.z) / 3.0f); // Calculated with "average distortion", ideal case: we move with the same amount in all directions
			currPos = vec4(pos.x, pos.y, pos.z, 1);
		}

		virtual void SetModellingTransform(mat4& M, mat4& Minv) {
			M = ScaleMatrix(scale) * RotationMatrix(rotAngle, rotAxis) * TranslateMatrix(pos) * RotationMatrix(rotAngle2, rotAxis2);
			Minv = RotationMatrix(-rotAngle2, rotAxis2) * TranslateMatrix(-pos) * RotationMatrix(-rotAngle, rotAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
		}

		virtual void Animate(float ts, float te) {
			if (move) {
				rotAngle = 0.8f * te;
				updateRotParams(te);
				geometry->updateTime(te);
				geometry->reCreate();

				mat4 pM, pMinv;
				SetModellingTransform(pM, pMinv);

				for (auto ch : children) {
					((Spike*)ch)->setParentMs(pM, pMinv);
					vec2 tuv = ((Spike*)ch)->texcoord;
					VertexData vtxd = ((ParamSurface*)geometry)->GenVertexData(tuv.x, tuv.y);
					((Spike*)ch)->pos = vtxd.position;
					((Spike*)ch)->n = vtxd.normal;
				}

				int nSpikesPerEq = 20;
				float r = 2.0f;
				float dt1 = 2.0f * M_PI / (float)nSpikesPerEq;

				// the current position of the virus can be obtained by transforming the previous position with the M matrix
				currPos = currPos * pM;
			}
		}

		vec3 getPos() { return vec3(currPos.x, currPos.y, currPos.z); }

		void stop() {
			move = false;
		}
	};


struct AntiBody : public Object {
		float tenthSec = 0;
		float control[3] = { 0.0f, 0.0f, 0.0f };
		float moveMultp = 0.14f;

		float R;

		AntiBody(Shader* shader_, Material* material_, Texture* texture_, Geometry* geometry_) : Object(shader_, material_, texture_, geometry_) {
			srand(0);
			pos = vec3(-2, 0, -2);
			scale = vec3(1.2f, 1.2f, 1.2f);
			R = geometry->radOfCircumSphere() * ((scale.x + scale.y + scale.z) / 3.0f); // "atlagos torzitassal" szamolva, idealis eset: minden iranyban ugyanannyival nyujtunk
		}

		virtual void Animate(float ts, float te) {
			rotAngle = 0.8f * te;
			geometry->updateTime(te);
			geometry->reCreate();

			// in this case, the mp count is unnecessary because the scene's animate() animates every 0.1s by default
			tenthSec += te - ts;
			if (tenthSec > 0.1f) {
				tenthSec -= 0.1f;

				float xR = (float)(((float)rand() / (float)RAND_MAX) - 0.5f + control[0]) * moveMultp;
				float yR = (float)(((float)rand() / (float)RAND_MAX) - 0.5f + control[1]) * moveMultp;
				float zR = (float)(((float)rand() / (float)RAND_MAX) - 0.5f + control[2]) * moveMultp;

				vec3 dpos(xR, yR, zR);
				pos = pos + dpos;
			}
		}

		// the antibody position is only affected by the displacement transformation, so it is enough to see this, but we can also multiply by the M matrix
		vec3 getPos() { return pos; }
	};


class Scene {
		Camera camera;
		std::vector<Object*> objects;
		std::vector<Light> lights;
		AntiBody* antiRef;
		Virus* virusRef;

	public:
		void Build() {
			// Shaders
			Shader* phongShader = new PhongShader();

			// Materials
			Material* mat0 = new Material;
			mat0->kd = vec3(0.6f, 0.7f, 0.2f);
			mat0->ka = vec3(0.3f, 0.2f, 0.3f);
			mat0->ks = vec3(2, 2, 2);
			mat0->shine = 200;

			Material* mat1 = new Material;
			mat1->kd = vec3(0.3f, 0.4f, 0.8f);
			mat1->ka = vec3(0.4f, 0.8f, 0.4f);
			mat1->ks = vec3(2, 2, 2);
			mat1->shine = 100;

			Material* mat2 = new Material;
			mat2->kd = vec3(0.1f, 0.2f, 0.3f);
			mat2->ka = vec3(0.1f, 0.2f, 0.2f);
			mat2->ks = vec3(2, 1, 2);
			mat2->shine = 500;

			Material* mat3 = new Material;
			mat3->kd = vec3(0.5f, 0.9f, 1.6f);
			mat3->ka = vec3(0.1f, 0.8f, 0.6f);
			mat3->ks = vec3(1, 3, 2);
			mat3->shine = 500;

			// Textures
			Texture* spikeText = new CheckedTexture(4, 8);
			Texture* bg = new CheckedTexture(10, 10);
			Texture* virusText = new StripedTexture(20, 20);

			// Geometries
			ParamSurface* virusBase = new VirusBase();
			Geometry* spikeBase = new Tractricoid();
			Geometry* gomb = new Sphere();
			Geometry* tetraBase = new recTetraeder();

			//Objects
			Object* room = new Object(phongShader, mat2, bg, gomb);
			room->scale = vec3(9, 9, 9);
			objects.push_back(room);

			Virus* virus = new Virus(phongShader, mat0, virusText, virusBase);
			virus->rotAxis = vec3(1, 0, 0);
			virus->rotAngle = 0.1f;
			objects.push_back(virus);
			virusRef = virus;

			int nSpikes = 20;
			float r = 2.0f;
			float dt1 = 2.0f * M_PI / (float)nSpikes;

			for (float fi = 0; fi < M_PI; fi += dt1) {
				int nCurr = (nSpikes * sinf(M_PI - fi)) / 1;
				for (int i = 0; i < nCurr; i++) {
					float theta = ((2.0f * M_PI) / (float)nCurr) * (float)i;

					VertexData vtxd = virusBase->GenVertexData(theta / (2.0f * M_PI), fi / M_PI);
					mat4 pM, pMinv;
					virus->SetModellingTransform(pM, pMinv);
					Spike* trc = new Spike(phongShader, mat1, spikeText, spikeBase, vtxd.normal, pM, pMinv);
					trc->scale = vec3(0.2, 0.2, 0.2);
					trc->pos = vtxd.position;
					trc->texcoord = vtxd.texcoord;
					virus->children.push_back(trc);
				}
			}

			AntiBody* tetra = new AntiBody(phongShader, mat3, spikeText, tetraBase);
			tetra->rotAxis = vec3(0, 1, 1);
			objects.push_back(tetra);
			antiRef = tetra;

			// Camera
			camera.wEye = vec3(0, 0, 8);
			camera.wLookat = vec3(0, 0, 0);
			camera.wVup = vec3(0, 1, 0);

			// Lights
			Light l1;
			l1.La = vec3(0.5f, 0.3f, 0.4f);
			l1.Le = vec3(0.6, 0.8, 0.9);
			l1.wLightPos = vec4(2, 2, 8, 1);
			lights.push_back(l1);
			Light l2;
			l2.La = vec3(0.3f, 0.3f, 0.5f);
			l2.Le = vec3(1, 1, 1.3);
			l2.wLightPos = vec4(-5, 5, -5, 1);
			lights.push_back(l2);
			Light l3;
			l3.La = vec3(0.7f, 0.7f, 0.7f);
			l3.Le = vec3(1.8, 1.8, 1.8);
			l3.wLightPos = vec4(0, -5, 6, 1);
			lights.push_back(l3);
		}

		void Render() {
			RenderState state;
			state.wEye = camera.wEye;
			state.V = camera.V();
			state.P = camera.P();
			state.lights = lights;
			for (Object* obj : objects)
				obj->Draw(state);
		}

		void Animate(float ts, float te) {
			for (Object* obj : objects) obj->Animate(ts, te);

			if (length(virusRef->getPos() - antiRef->getPos()) < (virusRef->R + antiRef->R))
				virusRef->stop();
		}

		void manipCoord(bool on, bool positive, int which) {
			if (on)
				antiRef->control[which] = (positive ? 1.0f : -1.0f) * (antiRef->moveMultp * 0.75f); // 75%-kal nagyobb es�ly
			else
				antiRef->control[which] = 0.0f;
		}
	};

Scene scene;

	// Initialization, create an OpenGL context
void onInitialization() {
		glViewport(0, 0, windowWidth, windowHeight);
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);
		scene.Build();
	}

	// Window has become invalid: Redraw
void onDisplay() {
		glClearColor(0.1f, 0.1f, 0, 0);     // background color
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear frame buffer
		scene.Render();
		glutSwapBuffers(); // exchange buffers for double buffering
	}

	// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
		// kontrollok, ha x,y,z lenyomodik valtoztatjuk az irany valoszinuseget
		if ((key >= 'x' && key <= 'z')) {
			scene.manipCoord(true, true, key - 'x');
		}
		else if (((key >= 'X' && key <= 'Z'))) {
			scene.manipCoord(true, false, key - 'X');
		}
	}

	// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
		if ((key >= 'x' && key <= 'z')) {
			scene.manipCoord(false, true, key - 'x');
		}
		else if (((key >= 'X' && key <= 'Z'))) {
			scene.manipCoord(false, false, key - 'X');
		}
	}

	// Move mouse with key pressed
	void onMouseMotion(int pX, int pY) {}

	// Mouse click event
	void onMouse(int button, int state, int pX, int pY) {}

	// Idle event indicating that some time elapsed: do animation here
	void onIdle() {
		static float tend = 0;
		const float dt = 0.1f; // dt is �infinitesimal�
		float tstart = tend;
		tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

		for (float t = tstart; t < tend; t += dt) {
			float Dt = fmin(dt, tend - t);
			scene.Animate(t, t + Dt);
		}

		glutPostRedisplay();
	}

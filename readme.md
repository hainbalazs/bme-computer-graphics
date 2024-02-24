# Computer graphics projects
In this repository there are two project listed below: Sunbeam simulation (raytracing), Antibody (animation). 
Each directory contains a code.cpp which shows the backbone of the project. If you want to build the entire project, please refer to the Programs library.
If you are interested about the results, there is a preview.png image included in each directory.

## Scope of the projects 
The following projects were part of a course work for Computer Graphics (VIIIAB07) at Budapest University of Technology and Economics, which I made during the 4. semester of my Bachelor's there.
 
## Overview
The aim of the projects are to introduce theory (projective geometry, color theory, analytic geometry, radiometry, kinetics/dynamics) in a practical way, using C++, the OpenGL 3 library and the GLSL shading programming language as implementation platform. It also helped me understand the architecture of 2D and 3D graphics systems, animation and game development techniques, and the operation and programming of graphics cards.

Required skills for the project: C++, OpenGL3, LSL Shading Language, Projective Geometry, Animations.

## Task descriptions
### Sunbeam simulation (raytracing)
Velux wants to test its solar tubes in a virtual world before installation. The task is yours for the case where the solar tube is a silver monocoque hyperboloid and the sun does not directly illuminate the room. In the room illuminated by the sunbeam, there are at least three objects, which can be freely chosen, but they cannot be spheres. At least one of the objects shall be optically smooth gold. The lumped materials follow the diffuse+PhongBlinn model. Sun+sky light is characterized by constant sky radiances and increasing solar radiances around the sun's direction. Sky light can only enter the room through the tube where only one ambient light source is present. The task is to photograph the scene from a virtual camera in the room.

![Sunbeam Simulator task preview](https://github.com/hainbalazs/bme-computer-graphics/blob/main/sunbeam-simulator/preview.png?raw=true)

### Antibody (animation)
Create an "antibody virus kills" game that takes place inside a textured sphere or cylinder, lit by point light sources. The body of the virus is an angrily undulating sphere, the extensions are Tractricoid shapes that are always perpendicular to the undulating surface. The projections cover the surface evenly. The sphere and projections are textured diffuse/specular type. The virus rotates about its own axis at a constant angular velocity, and also about a pivot point outside its body, given by the quaternion [cos(t), sin(t/2), sin(t/3), sin(t/5)] (caution: not normalized!) (t is time in sec). The antibody is a tetrahedron of Helge von Koch type with a two-level division. The antibody causes fear by stretching its spikes. The antibody rotates around its own axis and moves forward with Brownian motion, holding down the keys x, X, y, Y, z, Z, the progress is more likely in the given (lower case: positive, upper case: negative) direction. The velocity vector of Brownian motion is random and varies every 0.1 sec. If the sphere occupying the base tetrahedron of the antibody collides with the sphere of the base virus, the virus is destroyed, i.e. its movement ceases.

![Antibody task preview](https://github.com/hainbalazs/bme-computer-graphics/blob/main/antibody/preview.png?raw=true)

## Authorship
The entire project was developed by Balázs Hain including: design, graphics, implementation and documentation.
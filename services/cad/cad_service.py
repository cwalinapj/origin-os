#!/usr/bin/env python3
"""
Origin OS CAD Service - Generate STL files programmatically
Supports: numpy-stl, CadQuery, SolidPython/OpenSCAD, Trimesh
"""

import os
import json
import tempfile
import subprocess
import math
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="Origin OS CAD Service", version="1.0")

# Output directory for STL files
OUTPUT_DIR = os.getenv("CAD_OUTPUT_DIR", "/data/stl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# PRIMITIVE GENERATORS (numpy-stl based)
# =============================================================================

def create_cube(size: float = 1.0, center: Tuple[float, float, float] = (0, 0, 0)) -> 'mesh.Mesh':
    """Create a cube mesh"""
    from stl import mesh
    
    s = size / 2
    cx, cy, cz = center
    
    vertices = np.array([
        [cx-s, cy-s, cz-s], [cx+s, cy-s, cz-s], [cx+s, cy+s, cz-s], [cx-s, cy+s, cz-s],
        [cx-s, cy-s, cz+s], [cx+s, cy-s, cz+s], [cx+s, cy+s, cz+s], [cx-s, cy+s, cz+s]
    ])
    
    faces = np.array([
        [0,3,1], [1,3,2],  # bottom
        [0,4,7], [0,7,3],  # left
        [4,5,6], [4,6,7],  # top
        [5,1,2], [5,2,6],  # right
        [2,3,6], [3,7,6],  # front
        [0,1,5], [0,5,4]   # back
    ])
    
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[f[j], :]
    
    return cube


def create_box(width: float, height: float, depth: float, center: Tuple[float, float, float] = (0, 0, 0)) -> 'mesh.Mesh':
    """Create a rectangular box"""
    from stl import mesh
    
    w, h, d = width/2, height/2, depth/2
    cx, cy, cz = center
    
    vertices = np.array([
        [cx-w, cy-h, cz-d], [cx+w, cy-h, cz-d], [cx+w, cy+h, cz-d], [cx-w, cy+h, cz-d],
        [cx-w, cy-h, cz+d], [cx+w, cy-h, cz+d], [cx+w, cy+h, cz+d], [cx-w, cy+h, cz+d]
    ])
    
    faces = np.array([
        [0,3,1], [1,3,2], [0,4,7], [0,7,3], [4,5,6], [4,6,7],
        [5,1,2], [5,2,6], [2,3,6], [3,7,6], [0,1,5], [0,5,4]
    ])
    
    box = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            box.vectors[i][j] = vertices[f[j], :]
    
    return box


def create_cylinder(radius: float, height: float, segments: int = 32, center: Tuple[float, float, float] = (0, 0, 0)) -> 'mesh.Mesh':
    """Create a cylinder mesh"""
    from stl import mesh
    
    cx, cy, cz = center
    h = height / 2
    
    # Generate vertices
    vertices = []
    # Bottom center
    vertices.append([cx, cy, cz - h])
    # Top center
    vertices.append([cx, cy, cz + h])
    
    # Bottom and top rings
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        vertices.append([x, y, cz - h])  # bottom ring
        vertices.append([x, y, cz + h])   # top ring
    
    vertices = np.array(vertices)
    
    # Generate faces
    faces = []
    for i in range(segments):
        # Bottom cap
        next_i = (i + 1) % segments
        faces.append([0, 2 + i*2, 2 + next_i*2])
        # Top cap
        faces.append([1, 3 + next_i*2, 3 + i*2])
        # Side faces
        faces.append([2 + i*2, 3 + i*2, 2 + next_i*2])
        faces.append([2 + next_i*2, 3 + i*2, 3 + next_i*2])
    
    faces = np.array(faces)
    
    cyl = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cyl.vectors[i][j] = vertices[f[j], :]
    
    return cyl


def create_sphere(radius: float, segments: int = 16, rings: int = 16, center: Tuple[float, float, float] = (0, 0, 0)) -> 'mesh.Mesh':
    """Create a sphere mesh using UV sphere method"""
    from stl import mesh
    
    cx, cy, cz = center
    vertices = []
    
    # Generate vertices
    for i in range(rings + 1):
        phi = math.pi * i / rings
        for j in range(segments):
            theta = 2 * math.pi * j / segments
            x = cx + radius * math.sin(phi) * math.cos(theta)
            y = cy + radius * math.sin(phi) * math.sin(theta)
            z = cz + radius * math.cos(phi)
            vertices.append([x, y, z])
    
    vertices = np.array(vertices)
    
    # Generate faces
    faces = []
    for i in range(rings):
        for j in range(segments):
            next_j = (j + 1) % segments
            v1 = i * segments + j
            v2 = i * segments + next_j
            v3 = (i + 1) * segments + j
            v4 = (i + 1) * segments + next_j
            
            if i != 0:
                faces.append([v1, v3, v2])
            if i != rings - 1:
                faces.append([v2, v3, v4])
    
    faces = np.array(faces)
    
    sphere = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            sphere.vectors[i][j] = vertices[f[j], :]
    
    return sphere


def create_cone(radius: float, height: float, segments: int = 32, center: Tuple[float, float, float] = (0, 0, 0)) -> 'mesh.Mesh':
    """Create a cone mesh"""
    from stl import mesh
    
    cx, cy, cz = center
    
    vertices = [[cx, cy, cz + height]]  # Apex
    vertices.append([cx, cy, cz])  # Base center
    
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        vertices.append([x, y, cz])
    
    vertices = np.array(vertices)
    
    faces = []
    for i in range(segments):
        next_i = (i + 1) % segments
        # Side face
        faces.append([0, 2 + i, 2 + next_i])
        # Base face
        faces.append([1, 2 + next_i, 2 + i])
    
    faces = np.array(faces)
    
    cone = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cone.vectors[i][j] = vertices[f[j], :]
    
    return cone


def create_torus(major_radius: float, minor_radius: float, major_segments: int = 32, minor_segments: int = 16, center: Tuple[float, float, float] = (0, 0, 0)) -> 'mesh.Mesh':
    """Create a torus mesh"""
    from stl import mesh
    
    cx, cy, cz = center
    vertices = []
    
    for i in range(major_segments):
        theta = 2 * math.pi * i / major_segments
        for j in range(minor_segments):
            phi = 2 * math.pi * j / minor_segments
            x = cx + (major_radius + minor_radius * math.cos(phi)) * math.cos(theta)
            y = cy + (major_radius + minor_radius * math.cos(phi)) * math.sin(theta)
            z = cz + minor_radius * math.sin(phi)
            vertices.append([x, y, z])
    
    vertices = np.array(vertices)
    
    faces = []
    for i in range(major_segments):
        next_i = (i + 1) % major_segments
        for j in range(minor_segments):
            next_j = (j + 1) % minor_segments
            v1 = i * minor_segments + j
            v2 = i * minor_segments + next_j
            v3 = next_i * minor_segments + j
            v4 = next_i * minor_segments + next_j
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])
    
    faces = np.array(faces)
    
    torus = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            torus.vectors[i][j] = vertices[f[j], :]
    
    return torus


def create_pyramid(base_size: float, height: float, center: Tuple[float, float, float] = (0, 0, 0)) -> 'mesh.Mesh':
    """Create a square pyramid"""
    from stl import mesh
    
    s = base_size / 2
    cx, cy, cz = center
    
    vertices = np.array([
        [cx, cy, cz + height],      # apex
        [cx-s, cy-s, cz],           # base corners
        [cx+s, cy-s, cz],
        [cx+s, cy+s, cz],
        [cx-s, cy+s, cz]
    ])
    
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],  # sides
        [1, 4, 2], [2, 4, 3]  # base
    ])
    
    pyramid = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            pyramid.vectors[i][j] = vertices[f[j], :]
    
    return pyramid


def create_text_3d(text: str, height: float = 10, depth: float = 2, font_size: float = 12) -> str:
    """Create 3D text using OpenSCAD (requires OpenSCAD installed)"""
    scad_code = f'''
    linear_extrude(height = {depth}) {{
        text("{text}", size = {font_size}, font = "Liberation Sans", halign = "center", valign = "center");
    }}
    '''
    
    # Write SCAD file
    scad_file = os.path.join(OUTPUT_DIR, f"text_{text[:10].replace(' ', '_')}.scad")
    stl_file = scad_file.replace('.scad', '.stl')
    
    with open(scad_file, 'w') as f:
        f.write(scad_code)
    
    # Run OpenSCAD to generate STL
    try:
        subprocess.run(['openscad', '-o', stl_file, scad_file], check=True, timeout=60)
        return stl_file
    except FileNotFoundError:
        raise HTTPException(500, "OpenSCAD not installed. Install with: brew install openscad")
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"OpenSCAD error: {e}")


def combine_meshes(meshes: List['mesh.Mesh']) -> 'mesh.Mesh':
    """Combine multiple meshes into one"""
    from stl import mesh
    
    combined = mesh.Mesh(np.concatenate([m.data for m in meshes]))
    return combined


# =============================================================================
# CADQUERY SUPPORT (more powerful CAD operations)
# =============================================================================

def cadquery_to_stl(code: str, filename: str) -> str:
    """Execute CadQuery code and export to STL"""
    try:
        import cadquery as cq
        
        # Execute the code
        local_vars = {'cq': cq}
        exec(code, local_vars)
        
        # Find the result object
        result = local_vars.get('result')
        if result is None:
            raise ValueError("CadQuery code must define a 'result' variable")
        
        # Export to STL
        filepath = os.path.join(OUTPUT_DIR, filename)
        cq.exporters.export(result, filepath)
        return filepath
    
    except ImportError:
        raise HTTPException(500, "CadQuery not installed. Install with: pip install cadquery")


# =============================================================================
# OPENSCAD SUPPORT
# =============================================================================

def openscad_to_stl(code: str, filename: str) -> str:
    """Render OpenSCAD code to STL"""
    scad_file = os.path.join(OUTPUT_DIR, filename.replace('.stl', '.scad'))
    stl_file = os.path.join(OUTPUT_DIR, filename)
    
    with open(scad_file, 'w') as f:
        f.write(code)
    
    try:
        subprocess.run(['openscad', '-o', stl_file, scad_file], check=True, timeout=300)
        return stl_file
    except FileNotFoundError:
        raise HTTPException(500, "OpenSCAD not installed")
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"OpenSCAD render error: {e}")


# =============================================================================
# API MODELS
# =============================================================================

class PrimitiveRequest(BaseModel):
    shape: str  # cube, box, cylinder, sphere, cone, torus, pyramid
    params: Dict[str, Any] = {}
    filename: Optional[str] = None


class CustomMeshRequest(BaseModel):
    vertices: List[List[float]]
    faces: List[List[int]]
    filename: Optional[str] = None


class CadQueryRequest(BaseModel):
    code: str
    filename: Optional[str] = None


class OpenSCADRequest(BaseModel):
    code: str
    filename: Optional[str] = None


class TextRequest(BaseModel):
    text: str
    height: float = 10
    depth: float = 2
    font_size: float = 12


class CombineRequest(BaseModel):
    shapes: List[PrimitiveRequest]
    filename: Optional[str] = None


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    return {
        "service": "Origin OS CAD Service",
        "version": "1.0",
        "capabilities": {
            "primitives": ["cube", "box", "cylinder", "sphere", "cone", "torus", "pyramid"],
            "custom": "Create from vertices and faces",
            "cadquery": "Full CadQuery support",
            "openscad": "OpenSCAD code rendering",
            "text": "3D text generation",
            "combine": "Combine multiple shapes"
        },
        "endpoints": {
            "primitive": "POST /primitive - Create primitive shape",
            "custom": "POST /custom - Create from vertices/faces",
            "cadquery": "POST /cadquery - Run CadQuery code",
            "openscad": "POST /openscad - Run OpenSCAD code",
            "text": "POST /text - Create 3D text",
            "combine": "POST /combine - Combine shapes",
            "download": "GET /download/{filename} - Download STL",
            "list": "GET /list - List generated files"
        }
    }


@app.post("/primitive")
async def create_primitive(req: PrimitiveRequest):
    """Create a primitive shape STL"""
    from stl import mesh as stl_mesh
    
    shape = req.shape.lower()
    params = req.params
    filename = req.filename or f"{shape}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.stl"
    
    generators = {
        "cube": lambda: create_cube(params.get("size", 1), tuple(params.get("center", [0, 0, 0]))),
        "box": lambda: create_box(params.get("width", 1), params.get("height", 1), params.get("depth", 1), tuple(params.get("center", [0, 0, 0]))),
        "cylinder": lambda: create_cylinder(params.get("radius", 1), params.get("height", 2), params.get("segments", 32), tuple(params.get("center", [0, 0, 0]))),
        "sphere": lambda: create_sphere(params.get("radius", 1), params.get("segments", 16), params.get("rings", 16), tuple(params.get("center", [0, 0, 0]))),
        "cone": lambda: create_cone(params.get("radius", 1), params.get("height", 2), params.get("segments", 32), tuple(params.get("center", [0, 0, 0]))),
        "torus": lambda: create_torus(params.get("major_radius", 2), params.get("minor_radius", 0.5), params.get("major_segments", 32), params.get("minor_segments", 16), tuple(params.get("center", [0, 0, 0]))),
        "pyramid": lambda: create_pyramid(params.get("base_size", 2), params.get("height", 2), tuple(params.get("center", [0, 0, 0])))
    }
    
    if shape not in generators:
        raise HTTPException(400, f"Unknown shape: {shape}. Available: {list(generators.keys())}")
    
    mesh_obj = generators[shape]()
    filepath = os.path.join(OUTPUT_DIR, filename)
    mesh_obj.save(filepath)
    
    return {
        "success": True,
        "shape": shape,
        "filename": filename,
        "filepath": filepath,
        "download_url": f"/download/{filename}"
    }


@app.post("/custom")
async def create_custom(req: CustomMeshRequest):
    """Create STL from custom vertices and faces"""
    from stl import mesh as stl_mesh
    
    vertices = np.array(req.vertices)
    faces = np.array(req.faces)
    filename = req.filename or f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.stl"
    
    mesh_obj = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            mesh_obj.vectors[i][j] = vertices[f[j], :]
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    mesh_obj.save(filepath)
    
    return {
        "success": True,
        "filename": filename,
        "vertices": len(vertices),
        "faces": len(faces),
        "download_url": f"/download/{filename}"
    }


@app.post("/cadquery")
async def run_cadquery(req: CadQueryRequest):
    """Execute CadQuery code and generate STL"""
    filename = req.filename or f"cadquery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.stl"
    filepath = cadquery_to_stl(req.code, filename)
    
    return {
        "success": True,
        "filename": filename,
        "filepath": filepath,
        "download_url": f"/download/{filename}"
    }


@app.post("/openscad")
async def run_openscad(req: OpenSCADRequest):
    """Execute OpenSCAD code and generate STL"""
    filename = req.filename or f"openscad_{datetime.now().strftime('%Y%m%d_%H%M%S')}.stl"
    filepath = openscad_to_stl(req.code, filename)
    
    return {
        "success": True,
        "filename": filename,
        "filepath": filepath,
        "download_url": f"/download/{filename}"
    }


@app.post("/text")
async def create_text(req: TextRequest):
    """Create 3D text STL"""
    filepath = create_text_3d(req.text, req.height, req.depth, req.font_size)
    filename = os.path.basename(filepath)
    
    return {
        "success": True,
        "text": req.text,
        "filename": filename,
        "download_url": f"/download/{filename}"
    }


@app.post("/combine")
async def combine_shapes(req: CombineRequest):
    """Combine multiple primitive shapes into one STL"""
    from stl import mesh as stl_mesh
    
    meshes = []
    for shape_req in req.shapes:
        # Reuse primitive logic
        shape = shape_req.shape.lower()
        params = shape_req.params
        
        generators = {
            "cube": lambda p=params: create_cube(p.get("size", 1), tuple(p.get("center", [0, 0, 0]))),
            "box": lambda p=params: create_box(p.get("width", 1), p.get("height", 1), p.get("depth", 1), tuple(p.get("center", [0, 0, 0]))),
            "cylinder": lambda p=params: create_cylinder(p.get("radius", 1), p.get("height", 2), p.get("segments", 32), tuple(p.get("center", [0, 0, 0]))),
            "sphere": lambda p=params: create_sphere(p.get("radius", 1), p.get("segments", 16), p.get("rings", 16), tuple(p.get("center", [0, 0, 0]))),
            "cone": lambda p=params: create_cone(p.get("radius", 1), p.get("height", 2), p.get("segments", 32), tuple(p.get("center", [0, 0, 0]))),
            "pyramid": lambda p=params: create_pyramid(p.get("base_size", 2), p.get("height", 2), tuple(p.get("center", [0, 0, 0])))
        }
        
        if shape in generators:
            meshes.append(generators[shape]())
    
    if not meshes:
        raise HTTPException(400, "No valid shapes to combine")
    
    combined = combine_meshes(meshes)
    filename = req.filename or f"combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.stl"
    filepath = os.path.join(OUTPUT_DIR, filename)
    combined.save(filepath)
    
    return {
        "success": True,
        "shapes_combined": len(meshes),
        "filename": filename,
        "download_url": f"/download/{filename}"
    }


@app.get("/download/{filename}")
async def download_stl(filename: str):
    """Download a generated STL file"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(404, f"File not found: {filename}")
    
    return FileResponse(filepath, media_type="application/octet-stream", filename=filename)


@app.get("/list")
async def list_files():
    """List all generated STL files"""
    files = []
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.stl'):
            filepath = os.path.join(OUTPUT_DIR, f)
            stat = os.stat(filepath)
            files.append({
                "filename": f,
                "size_bytes": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "download_url": f"/download/{f}"
            })
    
    return {"files": files, "count": len(files)}


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "cad"}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🔧 ORIGIN OS CAD SERVICE")
    print("=" * 60)
    print("\nCapabilities:")
    print("  • Primitives: cube, box, cylinder, sphere, cone, torus, pyramid")
    print("  • Custom meshes from vertices/faces")
    print("  • CadQuery support (parametric CAD)")
    print("  • OpenSCAD support (CSG modeling)")
    print("  • 3D text generation")
    print("  • Shape combining")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

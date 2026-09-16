"""Optional local-deployment import acceptance test using the installed metadata probe.

Creates dedicated temporary targets; deletes only those targets and probe imports.
Uses credentials already configured inside the importer container, never host arguments.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
import uuid
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def container_python(container, code):
    result = subprocess.run(["docker","exec","-i",container,"python","-"],input=code,text=True,capture_output=True,check=True)
    return result.stdout.strip()


CONNECT = '''import os
from omero.gateway import BlitzGateway
from omero.model import DatasetI, ScreenI
from omero.rtypes import rstring
c=BlitzGateway(os.environ['OMERO_USER'],os.environ['OMERO_PASSWORD'],host=os.environ['OMERO_HOST'],port=int(os.environ.get('OMERO_PORT','4064')))
assert c.connect()
c.setGroupForSession(0)
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--probe",type=Path,default=ROOT.parent / "cideconvolve/tools/omero_import_metadata_probe/omero_import_metadata_probe.py")
    p.add_argument("--container",default="deployment_scenarios-biomero-importer-1")
    args=p.parse_args()
    if not args.input.is_dir() or not args.probe.is_file():
        raise FileNotFoundError("Expected an existing output store and installed metadata probe")
    # The deployment probe's original /tmp is not shared with OMERO's reader.
    source=args.probe.read_text(encoding="utf-8").replace("/tmp/cideconvolve_omero_probe/","/data/.cidenoise_import_smoke/")
    module=types.ModuleType("_cidenoise_site_probe")
    module.__file__=str(args.probe)
    sys.modules[module.__name__]=module
    exec(compile(source,str(args.probe),"exec"),module.__dict__)
    # Read actual pixels through OMERO before the probe cleans up its imports.
    original_helper = module._container_helper_code
    pixel_probe = '''
_original_summarize_image = summarize_image
def summarize_image(conn, image_id):
    import hashlib
    import numpy as np
    summary = _original_summarize_image(conn, image_id)
    image = conn.getObject('Image', int(image_id))
    plane = image.getPrimaryPixels().getPlane(0, 0, 0)
    summary['pixel_probe_sha256'] = hashlib.sha256(np.asarray(plane, dtype='<f4').tobytes()).hexdigest()
    return summary

'''
    def helper():
        text = original_helper()
        marker = "payload = json.load(sys.stdin)"
        if text.count(marker) != 1:
            raise RuntimeError("Site probe helper changed; cannot insert pixel validation")
        return text.replace(marker, pixel_probe + marker)
    module._container_helper_code = helper
    import zarr
    kind="Screen" if "plate" in zarr.open(str(args.input),mode="r").attrs else "Dataset"
    name="CIDenoise smoke " + uuid.uuid4().hex
    create=CONNECT + f"o={kind}I(); o.setName(rstring({name!r})); o=c.getUpdateService().saveAndReturnObject(o); print(o.getId().getValue()); c.close()\n"
    identifier=int(container_python(args.container,create).splitlines()[-1])
    args.out.mkdir(parents=True,exist_ok=True)
    try:
        result=module.main(["--input",str(args.input.resolve()),"--out",str(args.out.resolve()),
            "--importer-container",args.container,"--target",f"{kind}:{identifier}","--user","root","--group","system","--mode","both","--cleanup","always"])
        if result:
            raise RuntimeError(f"Import probe failed: {result}")
        from cidenoise.ome_zarr import open_images
        source_images = open_images(args.input)[1]
        expected = sorted(hashlib.sha256(np.asarray(image.plane(0,0,0),dtype='<f4').tobytes()).hexdigest() for image in source_images)
        report = json.loads((args.out / "report.json").read_text(encoding="utf-8"))
        checks = {}
        for mode in ("direct", "biomero"):
            imported = report["imports"][mode]
            objects = imported.get("objects", {})
            images = objects.get("images", []) + [image for plate in objects.get("plates", []) for image in plate["images"]]
            actual = sorted(image["pixel_probe_sha256"] for image in images)
            if actual != expected or imported.get("error") or imported.get("import_failed"):
                raise AssertionError(f"{mode}: imported pixels do not match output stores")
            checks[mode] = {"fields":len(images),"first_plane_per_field_exact":True}
        (args.out / "pixel_validation.json").write_text(json.dumps(checks,indent=2))
    finally:
        cleanup=CONNECT + f"c.deleteObjects({kind!r},[{identifier}],deleteAnns=True,deleteChildren=True,wait=True); c.close()\n"
        container_python(args.container,cleanup)
    print(f"Reports: {args.out}")


if __name__ == "__main__":
    main()

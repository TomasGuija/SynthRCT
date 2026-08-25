"""Render a sequence of binary masks as 3D Slicer surface screenshots."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import slicer
import vtk


def clear_scene() -> None:
    slicer.mrmlScene.Clear(False)


def load_segmentation(mask_path: str, common: dict):
    label_node = slicer.util.loadLabelVolume(mask_path)
    if label_node is None:
        raise RuntimeError(f"Could not load mask: {mask_path}")

    segmentation_node = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLSegmentationNode",
        "LungSegmentation",
    )

    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
        label_node,
        segmentation_node,
    )
    slicer.mrmlScene.RemoveNode(label_node)

    segmentation_node.CreateClosedSurfaceRepresentation()
    segmentation = segmentation_node.GetSegmentation()

    segment_ids = vtk.vtkStringArray()
    segmentation.GetSegmentIDs(segment_ids)

    append = vtk.vtkAppendPolyData()

    for index in range(segment_ids.GetNumberOfValues()):
        polydata = vtk.vtkPolyData()
        segmentation_node.GetClosedSurfaceRepresentation(
            segment_ids.GetValue(index),
            polydata,
        )
        append.AddInputData(polydata)

    append.Update()

    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputConnection(append.GetOutputPort())
    smooth.SetNumberOfIterations(common["smooth_iterations"])
    smooth.SetPassBand(common["pass_band"])
    smooth.BoundarySmoothingOff()
    smooth.FeatureEdgeSmoothingOff()
    smooth.NonManifoldSmoothingOn()
    smooth.NormalizeCoordinatesOn()
    smooth.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(smooth.GetOutputPort())
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()

    model_node = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLModelNode",
        "Lung",
    )
    model_node.SetAndObservePolyData(normals.GetOutput())
    model_node.CreateDefaultDisplayNodes()

    display = model_node.GetDisplayNode()
    display.SetColor(*common["color"])
    display.SetOpacity(common["opacity"])
    display.SetAmbient(common["ambient"])
    display.SetDiffuse(common["diffuse"])
    display.SetSpecular(common["specular"])
    display.SetPower(common["specular_power"])

    slicer.mrmlScene.RemoveNode(segmentation_node)
    return model_node


def get_bounds(segmentation_node) -> list[float]:
    bounds = [0.0] * 6
    segmentation_node.GetRASBounds(bounds)
    return bounds


def merge_bounds(bounds_list: list[list[float]]) -> list[float]:
    merged = bounds_list[0].copy()

    for bounds in bounds_list[1:]:
        merged[0] = min(merged[0], bounds[0])
        merged[1] = max(merged[1], bounds[1])
        merged[2] = min(merged[2], bounds[2])
        merged[3] = max(merged[3], bounds[3])
        merged[4] = min(merged[4], bounds[4])
        merged[5] = max(merged[5], bounds[5])

    return merged


def compute_shared_bounds(jobs: list[dict], common: dict) -> list[float]:
    bounds = []

    for job in jobs:
        clear_scene()
        node = load_segmentation(job["mask"], common)
        bounds.append(get_bounds(node))

    clear_scene()
    return merge_bounds(bounds)


def setup_view(common: dict):
    layout_manager = slicer.app.layoutManager()
    layout_manager.setLayout(
        slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView
    )

    widget = layout_manager.threeDWidget(0)
    view = widget.threeDView()

    width, height = common["width"], common["height"]

    widget.setFixedSize(width, height)
    view.setFixedSize(width, height)

    slicer.app.processEvents()
    view.renderWindow().SetSize(width, height)

    view_node = view.mrmlViewNode()
    view_node.SetBackgroundColor(*common["background"])
    view_node.SetBackgroundColor2(*common["background"])
    view_node.SetBoxVisible(False)
    view_node.SetAxisLabelsVisible(False)
    view_node.SetFiducialsVisible(False)
    view_node.SetRulerType(0)

    return view


def camera_position(
    center: list[float],
    radius: float,
    camera_view: str,
) -> tuple[list[float], list[float]]:
    if camera_view == "front":
        return (
            [center[0], center[1] + 3 * radius, center[2]],
            [0, 0, 1],
        )

    if camera_view == "back":
        return (
            [center[0], center[1] - 3 * radius, center[2]],
            [0, 0, 1],
        )

    if camera_view == "left":
        return (
            [center[0] - 3 * radius, center[1], center[2]],
            [0, 0, 1],
        )

    if camera_view == "right":
        return (
            [center[0] + 3 * radius, center[1], center[2]],
            [0, 0, 1],
        )

    if camera_view == "superior":
        return (
            [center[0], center[1], center[2] + 3 * radius],
            [0, 1, 0],
        )

    if camera_view == "inferior":
        return (
            [center[0], center[1], center[2] - 3 * radius],
            [0, 1, 0],
        )

    return (
        [
            center[0] - 0.8 * radius,
            center[1] + 2.8 * radius,
            center[2] - 0.7 * radius,
        ],
        [0, 0, 1],
    )


def configure_camera(
    view,
    bounds: list[float],
    common: dict,
) -> None:
    center = [
        (bounds[0] + bounds[1]) / 2,
        (bounds[2] + bounds[3]) / 2,
        (bounds[4] + bounds[5]) / 2,
    ]

    radius = max(
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
        1.0,
    )

    position, view_up = camera_position(
        center,
        radius,
        common["camera_view"],
    )

    camera = (
        slicer.modules.cameras.logic()
        .GetViewActiveCameraNode(view.mrmlViewNode())
        .GetCamera()
    )

    camera.SetPosition(*position)
    camera.SetFocalPoint(*center)
    camera.SetViewUp(*view_up)
    camera.ParallelProjectionOn()
    camera.SetParallelScale(radius / common["zoom"])
    camera.SetClippingRange(0.01 * radius, 10.0 * radius)

    slicer.app.processEvents()
    view.forceRender()


def save_png(view, output_path: str, scale: int = 1) -> None:
    slicer.util.forceRenderAllViews()

    render_window = view.renderWindow()
    render_window.Render()

    slicer.app.processEvents()
    view.forceRender()

    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(render_window)
    capture.SetInputBufferTypeToRGB()
    capture.ReadFrontBufferOff()
    capture.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(output_path)
    writer.SetInputConnection(capture.GetOutputPort())
    writer.Write()


def render_job(
    job: dict,
    common: dict,
    bounds: list[float],
) -> None:
    clear_scene()

    load_segmentation(job["mask"], common)
    view = setup_view(common)
    configure_camera(view, bounds, common)
    save_png(view, job["png"], common["scale"])


def main() -> None:
    payload_path = Path(sys.argv[-1])
    payload = json.loads(payload_path.read_text())

    common = payload["common"]
    jobs = payload["jobs"]
    bounds = compute_shared_bounds(jobs, common)

    for index, job in enumerate(jobs):
        render_job(job, common, bounds)
        print(f"[{index + 1}/{len(jobs)}] {job['png']}")

    slicer.app.quit()


if __name__ == "__main__":
    main()
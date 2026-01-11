from . import gpu


@gpu.processor
class RaycastProcessor:
    work: gpu.Buffer["Ray"]

    def main(self, gid: gpu.Vec3u):
        ray = self.work[gid.x]
        # Raycasting logic would go here
        pass


@gpu.struct
class Ray:
    origin: gpu.Vec3f
    direction: gpu.Vec3f

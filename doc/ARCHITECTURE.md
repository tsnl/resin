# Architecture

> [!WARNING]
> 
> WIP

We need a better way to organize this application. It's turning into spaghetti.

The key challenge is managing expensive resources like textures, geometry, etc. across
different modules.
-   Images
    -   Both draw_2d and draw_3d need images. Who caches them? Are they the same on the
        GPU?
    -   How flexible should images be? Some images don't need to go to the GPU at 
        all, others should be GPU-only, e.g. BC-compressed. Are all these the same image
        class with different features? Or different, incompatible classes?
-   Geometry
    -   Ray-tracer needs a BVH from geometry. A rasterizer (if we write one) does not.
        -   Who owns the BVH? Who caches it?

Recurring problems related to resources:
-   Ownership. Who caches? Lifetimes and resource disposal?
-   Dependent resources. Who caches them?

Proposed solution: resources are handles, module provides features, single global module.
-   `BaseResource` class underpins entire engine. `dispose_event: EventHub` broadcast.
    Dependent resources can subscribe to these messages.
-   Immutable => hashable as a key, stable identity.
-   Key insight: **resources are tightly coupled with resource managers**.

E.g. `class Image`
-   Each instance created with an `Image.get()`, which handles caching: 
    constructor should never be called directly.
-   Two options:
    -   `Image` monolith: have `image.py` handle _everything_
        -   `Image.load_file()`: caches loaded images
        -   Handles mapping onto the GPU, providing in multiple forms (e.g. one common
            GPU atlas).
    -   `Image` as an entity: have other modules use `Image` as a cache key for 
        accessing other resources.

In this way, the entire engine can break down into a handful of key resource classes 
that can be used across different modules: `Image`, `Geometry`, `Material`, etc.

> [!NOTE]
> 
> If we embrace this architecture, we could even write modules in different languages,
> exposing methods on the handle type. Think `diplomat` for binding Rust code.

> [!NOTE]
>
> This reminds me a lot of ECS.

Issues:
-   Cross-cutting concerns get lumped into one module. Related concerns get spread 
    across different modules.
    -   E.g. do we want BVH construction code to live in `geometry.py`? What about LOD
        management?
    -   Annoying that geometry loading isn't in the same place as a 3D renderer. It's
        great if we want to add another renderer using the same data, but...
-   Too granular for lifetime to be explicit.
    -   BVH depends on Geometry, Geometry depends on Scene, etc etc: it is hard to tell
        when things get cleaned up.
-   Too flexible: it's not obvious how stuff should be used.
    -   Do we really need dynamic texture atlases shared between the 2D and 3D 
        renderers?

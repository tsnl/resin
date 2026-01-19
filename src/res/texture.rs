//! Texture resources contain image data.

//
// API:
//

use crate::res::generic::{GenericManager, GenericResource};

pub type TextureManager = GenericManager<TextureBackend>;
pub type Texture = GenericResource<TextureBackend>;

//
// Implementation:
//

struct TextureBackend {}

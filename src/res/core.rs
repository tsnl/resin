//! Boilerplate to help implement resource managers and enforce standard interface patterns.

use parking_lot::RwLock;
use std::{
    error::Error,
    sync::{Arc, Weak},
};

pub struct GenericManager<B: GenericBackend> {
    backend: RwLock<B>,
}
impl<B: GenericBackend> GenericManager<B> {
    pub fn create<'a>(create_info: B::CreateInfo<'a>) -> Arc<Self> {
        let backend = B::new(create_info);
        Arc::new(Self {
            backend: RwLock::new(backend),
        })
    }
    pub fn add<'a>(
        self: &Arc<Self>,
        create_info: B::ResourceCreateInfo<'a>,
    ) -> Result<GenericResource<B>, B::ResourceCreateError> {
        let mut backend = self.backend.write();
        let resource_info = backend.add_impl(create_info)?;
        let resource = GenericResource::new(Arc::downgrade(self), resource_info);
        Ok(resource)
    }
    fn del(self: &Arc<Self>, resource: &B::ResourceInfo) {
        let mut backend = self.backend.write();
        backend.del_impl(&resource);
    }
}

pub struct GenericResource<B: GenericBackend> {
    manager: Weak<GenericManager<B>>,
    info: B::ResourceInfo,
}
impl<B: GenericBackend> GenericResource<B> {
    fn new(manager: Weak<GenericManager<B>>, info: B::ResourceInfo) -> Self {
        Self { manager, info }
    }
    pub fn info(&self) -> &B::ResourceInfo {
        &self.info
    }
}
impl<B: GenericBackend> Drop for GenericResource<B> {
    fn drop(&mut self) {
        if let Some(manager) = self.manager.upgrade() {
            manager.del(&self.info)
        }
    }
}

pub trait GenericBackend: Sized {
    type CreateInfo<'a>;
    type ResourceCreateInfo<'a>;
    type ResourceCreateError: Error;
    type ResourceInfo;

    fn new<'a>(create_info: Self::CreateInfo<'a>) -> Self;

    fn add_impl<'a>(
        &mut self,
        create_info: Self::ResourceCreateInfo<'a>,
    ) -> Result<Self::ResourceInfo, Self::ResourceCreateError>;

    fn del_impl(&mut self, resource: &Self::ResourceInfo);
}

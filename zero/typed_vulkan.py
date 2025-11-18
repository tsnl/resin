__all__ = [
    # Basic
    "vkGetInstanceProcAddr",
    "ffi",
    # Common
    "VkDeviceSize",
    "VkFlags",
    "VkSampleCountFlags",
    # Instance
    "VkInstance",
    "vkCreateInstance",
    "vkDestroyInstance",
    "VkApplicationInfo",
    "VkInstanceCreateInfo",
    "VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR",
    # Surface
    "VkSurfaceKHR",
    # Physical devices
    "VkPhysicalDevice",
    "vkEnumeratePhysicalDevices",
    "vkGetPhysicalDeviceProperties",
    "VkPhysicalDeviceProperties",
    "VkPhysicalDeviceLimits",
    "VK_PHYSICAL_DEVICE_TYPE_OTHER",
    "VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU",
    "VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
    "VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU",
    "VK_PHYSICAL_DEVICE_TYPE_CPU",
    # Physical device queues
    "vkGetPhysicalDeviceQueueFamilyProperties",
    "VkDeviceQueueCreateInfo",
    "VK_QUEUE_GRAPHICS_BIT",
    "VK_QUEUE_COMPUTE_BIT",
    "VK_QUEUE_TRANSFER_BIT",
    # Devices
    "VkDevice",
    "vkCreateDevice",
    "vkDestroyDevice",
    "VkDeviceCreateInfo",
]

from typing import TypeAlias

from vulkan import (
    # Basic
    vkGetInstanceProcAddr,
    ffi,
    # Instance:
    vkCreateInstance,
    vkDestroyInstance,
    VkInstanceCreateInfo,
    VkApplicationInfo,
    VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
    # Physical devices:
    vkEnumeratePhysicalDevices,
    vkGetPhysicalDeviceProperties,
    VkPhysicalDeviceProperties,
    VkPhysicalDeviceLimits,
    VK_PHYSICAL_DEVICE_TYPE_OTHER,
    VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
    VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU,
    VK_PHYSICAL_DEVICE_TYPE_CPU,
    # Surface: (N/A)
    # Queues:
    vkGetPhysicalDeviceQueueFamilyProperties,
    VkDeviceQueueCreateInfo,
    VK_QUEUE_GRAPHICS_BIT,
    VK_QUEUE_COMPUTE_BIT,
    VK_QUEUE_TRANSFER_BIT,
    # Devices:
    vkCreateDevice,
    vkDestroyDevice,
    VkDeviceCreateInfo,
    # Images:
    vkCreateImage,
    VkImageCreateInfo,
    vkDestroyImage,
    # Image views:
    vkCreateImageView,
    VkImageViewCreateInfo,
    vkDestroyImageView,
    # Samplers:
    vkCreateSampler,
    VkSamplerCreateInfo,
    vkDestroySampler,
)


class OpaqueResourceHandle:
    pass


VkDeviceSize: TypeAlias = int
VkFlags: TypeAlias = int
VkSampleCountFlags: TypeAlias = VkFlags


class VkInstance(OpaqueResourceHandle): ...


class VkPhysicalDevice(OpaqueResourceHandle): ...


class VkDevice(OpaqueResourceHandle): ...


class VkSurfaceKHR(OpaqueResourceHandle): ...

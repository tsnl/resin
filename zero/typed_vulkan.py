__all__ = [
    # Common
    "VkDeviceSize",
    "VkFlags",
    "VkSampleCountFlags",
    # Instance
    "VkInstance",
    "vkCreateInstance",
    "VkApplicationInfo",
    "VkInstanceCreateInfo",
    "vkEnumeratePhysicalDevices",
    # Physical devices
    "VkPhysicalDevice",
    "vkGetPhysicalDeviceProperties",
    "VkPhysicalDeviceProperties",
    "VkPhysicalDeviceLimits",
    # Devices
    "VkDevice",
]

from typing import Any, TypeAlias

from vulkan import (
    # Instance:
    vkCreateInstance,
    VkInstanceCreateInfo,
    VkApplicationInfo,
    vkEnumeratePhysicalDevices,
    # Physical devices:
    vkGetPhysicalDeviceProperties,
    VkPhysicalDeviceProperties,
    VkPhysicalDeviceLimits,
)


class OpaqueResourceHandle:
    pass


VkDeviceSize: TypeAlias = int
VkFlags: TypeAlias = int
VkSampleCountFlags: TypeAlias = VkFlags

VkInstance: TypeAlias = OpaqueResourceHandle

VkPhysicalDevice: TypeAlias = OpaqueResourceHandle

VkDevice: TypeAlias = OpaqueResourceHandle

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
    "VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR",
    # Physical devices
    "VkPhysicalDevice",
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
    "VkDeviceCreateInfo",
]

from typing import TypeAlias

from vulkan import (
    # Instance:
    vkCreateInstance,
    VkInstanceCreateInfo,
    VkApplicationInfo,
    vkEnumeratePhysicalDevices,
    VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
    # Physical devices:
    vkGetPhysicalDeviceProperties,
    VkPhysicalDeviceProperties,
    VkPhysicalDeviceLimits,
    VK_PHYSICAL_DEVICE_TYPE_OTHER,
    VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
    VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU,
    VK_PHYSICAL_DEVICE_TYPE_CPU,
    # Queues:
    vkGetPhysicalDeviceQueueFamilyProperties,
    VkDeviceQueueCreateInfo,
    VK_QUEUE_GRAPHICS_BIT,
    VK_QUEUE_COMPUTE_BIT,
    VK_QUEUE_TRANSFER_BIT,
    # Devices:
    vkCreateDevice,
    VkDeviceCreateInfo,
)


class OpaqueResourceHandle:
    pass


VkDeviceSize: TypeAlias = int
VkFlags: TypeAlias = int
VkSampleCountFlags: TypeAlias = VkFlags

VkInstance: TypeAlias = OpaqueResourceHandle
VkPhysicalDevice: TypeAlias = OpaqueResourceHandle
VkDevice: TypeAlias = OpaqueResourceHandle

"""Functions for working with Ray."""
import os
import ray


def initialize_ray():
    """
    Simple function to initialize ray. Checks enviornment for existence of "ip_head" for
    situations where the user is running on multiple nodes. Automatically creats a new
    ray cluster if it has not been initialized.
    """
    if not ray.is_initialized():
        print("Ray is not initialized. Trying with environment variables!")
        if os.environ.get("ip_head"):
            ray.init(
                address="auto",
                _node_ip_address=os.environ["ip_head"].split(":")[0],
                _redis_password=os.environ["redis_password"],
            )
        else:
            print("Could not identify existing Ray instance. Creating a new one...")
            ray.init(_redis_password="default_password")
        print(ray.nodes())


def to_iterator(obj_ids):
    """
    Method to convert a list of ray object ids to an iterator that can be used in a for loop.
    """
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])

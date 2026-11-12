"""Built-in feature backends; every module here is imported on discovery.

To add a backend, drop a module in this package (or expose one through the
``colmap_odometry.features`` entry point) that registers classes with
:func:`colmap_odometry.features.register`.  Import heavy frameworks inside
the classes, never at module level.
"""

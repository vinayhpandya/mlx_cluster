from setuptools import setup
from mlx import extension

if __name__ == "__main__":
    setup(
        name="mlx_graphs_extension",
        version="0.0.0",
        description="Sample C++ and Metal extensions for MLX primitives.",
        ext_modules=[extension.CMakeExtension("mlx_graphs_extension._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["mlx_graphs_extension"],
        package_data={"mlx_graphs_extension": ["*.so", "*.dylib", "*.metallib"]},
        extras_require={"dev": []},
        zip_safe=False,
        python_requires=">=3.8",
    )
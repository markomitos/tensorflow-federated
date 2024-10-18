load("@rules_license//rules:license.bzl", "license")
load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

package(
    default_applicable_licenses = [":package_license"],
    default_visibility = ["//visibility:public"],
)

license(
    name = "package_license",
    package_name = "tensorflow_federated",
    license_kinds = ["@rules_license//licenses/spdx:Apache-2.0"],
)

licenses(["notice"])

exports_files([
    "LICENSE",
    "README.md",
])

filegroup(
    name = "pyproject_toml",
    srcs = ["pyproject.toml"],
    visibility = ["//tools/python_package:python_package_tool"],
)

refresh_compile_commands(
    name = "refresh_compile_commands",

    targets = [
      "//tensorflow_federated/cc/...",
      "//tensorflow_federated/proto/..."
    ],
)
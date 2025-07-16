docker := require("docker")
rm := require("rm")
uv := require("uv")


PACKAGE := "fraudsys"
REPOSITORY := "fraudsys"
SOURCES := "src"
TESTS := "tests"

default:
  @just --list

import "tasks/check.just"
import "tasks/clean.just"
import "tasks/commit.just"
import "tasks/docker.just"
import "tasks/format.just"
import "tasks/infra.just"
import "tasks/install.just"
import "tasks/package.just"
import "tasks/pipeline.just"
import "tasks/project.just"
import "tasks/terraform.just"

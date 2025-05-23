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
import "tasks/package.just"
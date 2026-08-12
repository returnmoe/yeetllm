variable "YEETLLM_VERSION" {
  default = "0.1.0.dev0"
}

variable "SOURCE_REVISION" {
  default = "development"
}

group "default" {
  targets = ["image"]
}

target "image" {
  context    = "."
  dockerfile = "Dockerfile"
  platforms  = ["linux/amd64"]
  args = {
    YEETLLM_VERSION = YEETLLM_VERSION
    SOURCE_REVISION = SOURCE_REVISION
  }
  tags = ["yeetllm:development"]
}

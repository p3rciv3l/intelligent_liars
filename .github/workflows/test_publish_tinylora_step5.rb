# frozen_string_literal: true

require "yaml"

WORKFLOW_PATH = File.expand_path("publish-tinylora-step5.yml", __dir__)
WORKFLOW_TEXT = File.read(WORKFLOW_PATH)
WORKFLOW = YAML.safe_load(WORKFLOW_TEXT, aliases: true)

def assert_contract(condition, message)
  raise message unless condition
end

events = WORKFLOW["on"] || WORKFLOW[true] # Psych follows YAML 1.1 for `on`.
assert_contract(events.keys == ["workflow_dispatch"], "workflow must be manual-only")
assert_contract(
  WORKFLOW["permissions"] == {"contents" => "read", "packages" => "write"},
  "workflow permissions must stay minimal",
)

job = WORKFLOW.fetch("jobs").fetch("build-and-publish")
assert_contract(job.fetch("runs-on") == "ubuntu-24.04", "publication must not require a GPU")
assert_contract(
  job.fetch("env").fetch("IMAGE_REPOSITORY") ==
    "ghcr.io/p3rciv3l/intelligent_liars/tinylora-step5",
  "registry target changed",
)
assert_contract(
  job.fetch("env").fetch("IMAGE_TAG") == "sha-${{ github.sha }}",
  "tag must contain the full source commit",
)

steps = job.fetch("steps")
uses = steps.map { |step| step["uses"] }.compact
assert_contract(
  uses.all? { |action| action.match?(/@[0-9a-f]{40}$/) },
  "every third-party action must be pinned to a full commit",
)

build_step = steps.find { |step| step["id"] == "build" }
build = build_step.fetch("with")
assert_contract(build_step.fetch("if") == "steps.tag.outputs.should_build == 'true'", "existing tags must not be rebuilt")
assert_contract(build.fetch("push") == true, "image push must remain enabled")
assert_contract(build.fetch("platforms") == "linux/amd64", "platform changed")
assert_contract(build.fetch("provenance") == "mode=max", "max provenance required")
assert_contract(build.fetch("sbom") == true, "SBOM attestation required")
assert_contract(
  build.fetch("tags") == "${{ env.IMAGE_REPOSITORY }}:${{ env.IMAGE_TAG }}",
  "build must use the commit-addressed tag",
)

assert_contract(
  WORKFLOW_TEXT.scan(/secrets\.[A-Za-z0-9_]+/).uniq == ["secrets.GITHUB_TOKEN"],
  "only GITHUB_TOKEN may be referenced",
)
assert_contract(WORKFLOW_TEXT.include?("Registry lookup failed ambiguously"), "registry lookup must fail closed")
assert_contract(WORKFLOW_TEXT.include?("existing_revision"), "existing image revision must be verified")
assert_contract(WORKFLOW_TEXT.include?("tag_digest"), "tag digest must be verified after publication")

%w[image-digest.txt sbom.json provenance.json SHA256SUMS].each do |artifact|
  assert_contract(WORKFLOW_TEXT.include?(artifact), "missing publication evidence #{artifact}")
end

puts "publish workflow contract: pass"

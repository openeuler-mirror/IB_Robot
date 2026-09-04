# skill_catalog

`skill_catalog` is the SSOT for skill manifests, implementation bodies, and
profile definitions. Robot YAML in `robot_config` selects the active profile;
this package owns the resulting immutable Robot Skill Package catalog. Runtime
state, leases, task IDs, authorization, and action execution do not belong here.

## Layout

- `config/skills/<name>/manifest.yaml`: semantic description, public capability schema, implementation index.
- `config/skills/<name>/implementations/<implementation>.yaml`: robot-specific primitive or delegated execution body.
- `config/skills/<name>/SKILL.md`: human/Agent documentation only; never executable configuration.
- `config/profiles/<profile>.yaml`: enabled package, selected implementation, and planner visibility.

The current source-workspace profiles are `so101_single_arm`,
`lekiwi_handeye_realsense_grasp`, `lekiwi_handeye_realsense_grasp_pc`,
`lekiwi_handeye_realsense_grasp_lidar`, `so101_rtp_distributed`, and `lekiwi_lidar`.
The `lekiwi_handeye_realsense_grasp_lidar` profile is the unified mobile-manipulator
profile: it exposes manipulation, navigation, and approved social arm skills in one immutable snapshot.
The shared stable
implementation `so101_arm_v1` is selected by every enabled entry in both
`so101_single_arm` and `so101_rtp_distributed`; the two profiles compile the
same execution body, so their implementation content cannot drift through
duplicated files. Registry/capability identity can still differ through the
canonical robot execution context. `lekiwi_lidar` is the navigation-only
profile that selects the V2 schema contract (see [Context schema versions](#context-schema-versions)
below); it does not enable any manipulation primitive.

Both hand-eye profiles expose `pick_object` and `place_in_container` through
the Gateway. 310P binds grasping to the `ascend_310p` manifest deployment;
the PC profile binds it to the `torch_cuda` manifest deployment. The unified
`lekiwi_handeye_realsense_grasp_lidar` profile additionally exposes
`imitate_human_motion` through its launch-managed delegated executor.

## Source Modes

Robot YAML selects the source through `embodied.skill_catalog_source_mode`
(`installed`, `development`, or `production`), `embodied.skill_catalog_source_root`,
and `embodied.skill_catalog_profile`. Inline `embodied.skill_templates` is no
longer a valid execution SSOT; `robot_config.loader` rejects it with a single
error pointing at `embodied.skill_catalog_profile`.

- `development`: validates a mutable staging tree before and after compilation. Source-workspace robot configs use this
  mode with an absolute path resolved by `embodied_bringup`.
- `installed`: resolves `skill_catalog` through the ament package index. It rejects symlinked implementation files, so
  it is intended for non-`--symlink-install` release builds.
- `production`: requires a release directory selected through `current` and verified by release manifest/digest.

## Compilation integrity

Every non-hidden package directory under `config/skills/` must contain a
`manifest.yaml`. A malformed package directory (missing manifest, hidden
manifest, symlinked or escaped implementation path, forbidden release entry)
fails compilation with a `SKILL_SCHEMA_INVALID` / `SKILL_REFERENCE_MISSING` /
`SKILL_RELEASE_NOT_IMMUTABLE` diagnostic, even when the package is not enabled
in the active profile. Manifest schema validation also runs for packages
outside the enabled profile set, so catalog-wide drift is caught at compile
time rather than at execution time.

## Execution-context digest

`robot_config.loader.robot_config_digest` is the canonical execution-context
digest passed to `SkillRobotContext.robot_config_digest`. It deliberately
excludes catalog source selection (`skill_catalog_source_mode` /
`skill_catalog_source_root`), the active profile (`skill_catalog_profile`),
the resolved config path, and unrelated robot configuration. Flipping only
those fields keeps the same robot execution-context digest. It does not make
a complete snapshot reusable: profile remains part of registry/capability
identity and source release remains part of provenance identity. Changing named poses,
named targets, joint limits, workspace, control mode, timeout policy, or
execution endpoints produces a different digest and forces a new registry
generation.

## Consumer identity and generation sync

Every successful compile produces immutable registry, capability, and
provenance digests. Consumers must query an exact
`(registry_epoch, generation, registry_digest)` through Gateway services;
they must not read these files directly at validation or execution time.
`CatalogViewSynchronizer` never leaves a stale generation visible: when a
newer identity is announced, the previously verified view is cleared and the
consumer's exact identity becomes unavailable until the new snapshot has been
fetched and verified. Callers that cannot tolerate that gap must retain the
previous generation through the registry's retain/release API instead of
relying on the synchronizer's `current` view.

## Context schema versions

`SkillRobotContext.context_schema_version` selects the context contract used by
every consumer of the compiled snapshot. Three values are accepted:

- `1` (V1): the original manipulation-only contract. Skill and primitive
  requests must not carry navigation fields.
- `2` (V2): superset of V1 that adds the navigation field set
  (`direction` / `distance` / `degree` / `x` / `y` / `yaw`) plus
  `has_x` / `has_y` / `has_yaw` presence flags, and exposes three navigation
  primitives: `nav_straight`, `nav_turn`, `nav_abs_coordinate`.
- `3` (V3): hybrid context using the same V1+V2 primitive set while adding
  `supported_control_modes`. A single profile may contain skills whose capabilities
  require either `moveit_planning` or `base_navigation`; runtime ownership switching
  remains outside the catalog.

V3 is a context version, not a third public request wire version. A hybrid snapshot
accepts the existing V1 manipulation requests and V2 navigation requests in the same
catalog; each capability's `schema_version` still selects the corresponding request
fields and dispatch binding contract.

The active `context_schema_version` is derived from the resolved `robot_config` stage
and `navigation_endpoint_projection`: the `hybrid` stage resolves to V3, a standalone
navigation endpoint resolves to V2, and other stages resolve to V1. The
resolved version enters the canonical execution-context digest preimage, so
flipping it forces a new registry generation — V1, V2 and V3 snapshots are never
interchangeably validated.

`base_navigation` is the navigation control mode that authorizes the chassis velocity
controllers and grants `/cmd_vel` ownership to the navigation stack. It is the
only control mode under which `nav_*` primitives may be dispatched. V2 profiles
select it statically. V3 profiles declare both it and `moveit_planning`; the runtime
switches to the capability's required mode before dispatch and rejects the skill with
`CONTROL_MODE_MISMATCH` if the transition cannot be confirmed.

## Primitive contract versioning

`embodied_common` exposes `primitive_contract_for_version(version)` and the
three contract objects `PRIMITIVE_CONTRACT_V1` / `PRIMITIVE_CONTRACT_V2` /
`PRIMITIVE_CONTRACT_V3`.
`skill_catalog` resolves the matching digest per snapshot using the snapshot's
`context_schema_version`, and writes the result into the immutable
`primitive_contract_digest` field. The contract is therefore
**per-context-version**, not a single global constant: a V1 snapshot is
validated against `PRIMITIVE_CONTRACT_V1`, a V2 snapshot against
`PRIMITIVE_CONTRACT_V2`, and a V3 snapshot against `PRIMITIVE_CONTRACT_V3`.
Downstream consumers (`safety_guard`,
`skill_library`) must call `primitive_contract_for_version()` with the
snapshot's context version when re-computing the digest locally — comparing
a V2 snapshot against a V1 digest returns `SKILL_SNAPSHOT_DIGEST_MISMATCH`
and discards the snapshot.

A `PrimitiveContractSet` is also exported for callers that need the
versioned digest and descriptors together (e.g. tooling that displays versions side by
side). Production code paths must select exactly one version per snapshot;
they must not mix V1, V2 and V3 digests within a single validation request.

## Execution endpoint roles

The canonical execution context exposes a fixed set of execution endpoint
roles. V1 declares 10 roles: `arm_trajectory_action_name`,
`task_executor_action_name`, `pick_action_name`, `move_configuration_service`,
`ee_pose_topic`, `joint_state_topic`, `validate_skill_service`,
`validate_primitive_service`, `skill_gateway_status_service`, and
`skill_catalog_reload_service`. `skill_catalog_snapshot_service` and
`skill_registry_event_topic` are runtime plumbing and not part of the
execution-context digest.

V2 and V3 add `navigation_action_name` as an 11th role. The value is the sole
projection of `robot_config.navigation.command_server.action_name` (default
`/navigation/execute`); no other field may override it. Profiles that resolve
to V1 leave `navigation_action_name` empty, and the loader rejects any V1
request that declares a non-empty navigation action name. V2/V3 profiles with
an empty `navigation_action_name` fail compilation with
`SKILL_SCHEMA_INVALID`.

## SkillRobotContext.context_schema_version

`SkillRobotContext` is the canonical robot execution context structure
passed to `SkillCatalogCompiler`. It carries:

- `robot_config_digest`: the canonical execution-context digest from
  `robot_config.loader`, excluding catalog source selection fields.
- `context_schema_version`: `1`, `2` or `3`, resolved from the selected stage and
  `navigation_endpoint_projection`. Determines which
  `primitive_contract_digest` is selected and which wire fields are
  permitted on requests.
- `named_poses` / `named_targets` / `workspace_limits` /
  `arm_joint_names` / `joint_limits` / `timeout_policy`: the V1 manipulation
  context.
- `execution_endpoints`: the V1 10-role endpoint map, plus
  `navigation_action_name` when `context_schema_version >= 2`.
- `supported_control_modes`: present for V3 and lists the mutually-exclusive
  runtime domains selected by the skill executor.
- `control_mode`: the active startup control mode. For V2 snapshots this must be
  `base_navigation` (or another mode explicitly authorized for navigation
  primitives). V3 snapshots additionally carry `supported_control_modes`; the
  executor switches to each capability's required mode before dispatch.

Changing any of the above (other than catalog source selection) produces a
different `robot_config_digest` and forces a new registry generation.

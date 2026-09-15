cmake_minimum_required(VERSION 3.10)
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()
project(ibrobot_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(action_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Generate interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Variant.msg"
  "msg/VariantsList.msg"
  "msg/InferenceError.msg"
  "msg/InferenceFeatureSummary.msg"
  "msg/InferencePolicySummary.msg"
  "msg/InferencePipelineStatus.msg"
  "msg/VideoStreamDescriptor.msg"
  "msg/VideoStreamStatus.msg"
  "msg/DistributedInferenceRequest.msg"
  "msg/DistributedInferenceResult.msg"
  "msg/InferenceOutcome.msg"
  "msg/ScheduledInferenceError.msg"
  "msg/InferenceWorkCapacity.msg"
  "msg/InferenceServingStatus.msg"
  "msg/RobotStatus.msg"
  "msg/SkillCapabilityStatus.msg"
  "msg/TaskStep.msg"
  "msg/TaskCommand.msg"
  "msg/TaskStatus.msg"
  "msg/SkillDiagnostic.msg"
  "msg/TaskBudget.msg"
  "msg/DispatchBinding.msg"
  "msg/DelegatedExecutorIdentity.msg"
  "msg/WorkflowStep.msg"
  "msg/SkillRegistryEvent.msg"
  "msg/AgentPlan.msg"
  "msg/SceneAnalysisRequest.msg"
  "msg/SceneAnalysisResult.msg"
  "msg/VisualGameEvent.msg"
  "msg/SceneObject.msg"
  "msg/SceneObservation.msg"
  "msg/AttentionWeights.msg"
  "msg/JointCurrent.msg"
  "msg/Detection2D.msg"
  "msg/DetectionArray.msg"
  "msg/ModelRuntimeInfo.msg"
  "msg/MaskEmbedding.msg"
  "msg/TextEmbedding.msg"
  "msg/TrackState.msg"
  "msg/SemanticMapMetadata.msg"
  "msg/SemanticObject3D.msg"
  "msg/SemanticObject3DArray.msg"
  "msg/SynthesizedAudio.msg"
  "msg/GraspCandidate.msg"
  "msg/GraspCandidateArray.msg"
  "msg/SpeechDirection.msg"
  "msg/MHandProFrame.msg"
  "msg/HumanHandState.msg"
  "msg/StampedFloat32MultiArray.msg"
  "action/RecordEpisode.action"
  "action/RunPolicy.action"
  "action/DispatchInfer.action"
  "action/OpenInferenceSession.action"
  "action/ScheduledDispatchInfer.action"
  "action/CloseInferenceSession.action"
  "action/OpenPipelineBinding.action"
  "action/DispatchPipelineBinding.action"
  "action/ClosePipelineBinding.action"
  "action/ExecuteTaskPlan.action"
  "action/SkillCommand.action"
  "action/PrimitiveCommand.action"
  "action/PickObject.action"
  "action/PlaceObject.action"
  "action/ImitateHumanMotion.action"
  "action/ArmReturnHome.action"
  "action/ExecuteAgentPlan.action"
  "action/ExecuteNavigation.action"
  "srv/RecognizeFile.srv"
  "srv/SetHotwords.srv"
  "srv/MoveToConfiguration.srv"
  "srv/MoveToPose.srv"
  "srv/ValidateSkill.srv"
  "srv/ValidatePrimitive.srv"
  "srv/GetSkillGatewayStatus.srv"
  "srv/ReloadSkillCatalog.srv"
  "srv/GetSkillSnapshot.srv"
  "srv/BeginWorkflowExecution.srv"
  "srv/FinalizeWorkflowExecution.srv"
  "srv/PlanAgentCommand.srv"
  "srv/ValidateAgentPlan.srv"
  "srv/ConfirmAgentPlan.srv"
  "srv/StartVisualGame.srv"
  "srv/GetVisualGameResult.srv"
  "srv/EchoModel.srv"
  "srv/GenerateMasks.srv"
  "srv/GenerateGrasps.srv"
  "srv/RecognizeTags.srv"
  "srv/EncodeEmbeddings.srv"
  "srv/EncodeText.srv"
  "srv/DetectSegment.srv"
  "srv/GroundingDetect.srv"
  "srv/SegmentDetections.srv"
  "srv/PlanGrasp.srv"
  "srv/VerifyGrasp.srv"
  "srv/GetSemanticObjects.srv"
  "srv/ResolveSemanticTarget.srv"
  "srv/SynthesizeSpeech.srv"
  "srv/PlayAudioFile.srv"
  "srv/StartTracking.srv"
  "srv/StopTracking.srv"
  "srv/GetTrackState.srv"
  "srv/SetSoundFollowing.srv"
  "srv/ResetBenchmark.srv"
  "srv/StepBenchmark.srv"
  "srv/PreparePolicyEpisode.srv"
  "srv/GetBenchmarkPlan.srv"
  "srv/FinalizeBenchmarkScope.srv"
  DEPENDENCIES std_msgs action_msgs builtin_interfaces geometry_msgs sensor_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

/*
Copyright 2026 The Kubeflow Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package torch

import (
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	corev1ac "k8s.io/client-go/applyconfigurations/core/v1"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	"github.com/kubeflow/trainer/v2/pkg/constants"
	"github.com/kubeflow/trainer/v2/pkg/runtime"
	"github.com/kubeflow/trainer/v2/pkg/runtime/framework"
	utiltesting "github.com/kubeflow/trainer/v2/pkg/util/testing"
)

func TestLlamaFactoryEnforceMLPolicy(t *testing.T) {
	cases := map[string]struct {
		info              *runtime.Info
		trainJob          *trainer.TrainJob
		wantInfo          *runtime.Info
		wantMLPolicyError error
	}{
		"single-node llamafactory with 1 GPU": {
			trainJob: utiltesting.MakeTrainJobWrapper(metav1.NamespaceDefault, "lf-job").
				Trainer(
					utiltesting.MakeTrainJobTrainerWrapper().
						NumNodes(1).
						Container(
							"ghcr.io/kubeflow/trainer/llamafactory-trainer",
							[]string{"llamafactory-cli", "train"},
							[]string{"/config/training.yaml"},
							corev1.ResourceList{
								"nvidia.com/gpu": resource.MustParse("1"),
							},
						).
						Obj(),
				).
				RuntimeRef(
					trainer.SchemeGroupVersion.WithKind(trainer.ClusterTrainingRuntimeKind),
					"llamafactory-lora-sft",
				).
				Obj(),
			info: runtime.NewInfo(
				runtime.WithMLPolicySource(
					utiltesting.MakeMLPolicyWrapper().
						WithMLPolicySource(*utiltesting.MakeMLPolicySourceWrapper().
							TorchPolicy().
							Obj(),
						).
						Obj(),
				),
				runtime.WithPodSet(constants.Node, ptr.To(constants.AncestorTrainer), 1, corev1.PodSpec{}, corev1ac.PodSpec().
					WithContainers(corev1ac.Container().WithName(constants.Node)),
				),
			),
			wantInfo: &runtime.Info{
				Labels:      make(map[string]string),
				Annotations: make(map[string]string),
				RuntimePolicy: runtime.RuntimePolicy{
					MLPolicySource: utiltesting.MakeMLPolicySourceWrapper().
						TorchPolicy().
						Obj(),
				},
				TemplateSpec: runtime.TemplateSpec{
					PodSets: []runtime.PodSet{{
						Name:              constants.Node,
						Ancestor:          ptr.To(constants.AncestorTrainer),
						Count:             ptr.To[int32](1),
						SinglePodRequests: make(corev1.ResourceList),
						Containers: []runtime.Container{{
							Name: constants.Node,
							Env: []corev1ac.EnvVarApplyConfiguration{
								{
									Name:  ptr.To(constants.TorchEnvNumNodes),
									Value: ptr.To("1"),
								},
								{
									Name:  ptr.To(constants.TorchEnvNumProcPerNode),
									Value: ptr.To("auto"),
								},
								{
									Name: ptr.To(constants.TorchEnvNodeRank),
									ValueFrom: &corev1ac.EnvVarSourceApplyConfiguration{
										FieldRef: &corev1ac.ObjectFieldSelectorApplyConfiguration{
											FieldPath: ptr.To(constants.JobCompletionIndexFieldPath),
										},
									},
								},
								{
									Name:  ptr.To(constants.TorchEnvMasterAddr),
									Value: ptr.To("lf-job-node-0-0.lf-job"),
								},
								{
									Name:  ptr.To(constants.TorchEnvMasterPort),
									Value: ptr.To(fmt.Sprintf("%d", constants.ContainerTrainerPort)),
								},
								{
									Name:  ptr.To(constants.LlamaFactoryEnvNumNodes),
									Value: ptr.To("1"),
								},
								{
									Name:  ptr.To(constants.LlamaFactoryEnvNumProcPerNode),
									Value: ptr.To("auto"),
								},
								{
									Name: ptr.To(constants.LlamaFactoryEnvNodeRank),
									ValueFrom: &corev1ac.EnvVarSourceApplyConfiguration{
										FieldRef: &corev1ac.ObjectFieldSelectorApplyConfiguration{
											FieldPath: ptr.To(constants.JobCompletionIndexFieldPath),
										},
									},
								},
								{
									Name:  ptr.To(constants.LlamaFactoryEnvMasterAddr),
									Value: ptr.To("lf-job-node-0-0.lf-job"),
								},
								{
									Name:  ptr.To(constants.LlamaFactoryEnvMasterPort),
									Value: ptr.To(fmt.Sprintf("%d", constants.ContainerTrainerPort)),
								},
								{
									Name:  ptr.To(constants.LlamaFactoryEnvForceTorchrun),
									Value: ptr.To("1"),
								},
							},
							Ports: []corev1ac.ContainerPortApplyConfiguration{{
								ContainerPort: ptr.To[int32](constants.ContainerTrainerPort),
							}},
						}},
					}},
				},
				Scheduler: &runtime.Scheduler{PodLabels: make(map[string]string)},
			},
		},
		"multi-node llamafactory with 2 nodes": {
			trainJob: utiltesting.MakeTrainJobWrapper(metav1.NamespaceDefault, "lf-multi").
				Trainer(
					utiltesting.MakeTrainJobTrainerWrapper().
						NumNodes(2).
						Container(
							"ghcr.io/kubeflow/trainer/llamafactory-trainer",
							[]string{"llamafactory-cli", "train"},
							[]string{"/config/training.yaml"},
							corev1.ResourceList{
								"nvidia.com/gpu": resource.MustParse("1"),
							},
						).
						Obj(),
				).
				RuntimeRef(
					trainer.SchemeGroupVersion.WithKind(trainer.ClusterTrainingRuntimeKind),
					"llamafactory-lora-sft",
				).
				Obj(),
			info: runtime.NewInfo(
				runtime.WithMLPolicySource(
					utiltesting.MakeMLPolicyWrapper().
						WithMLPolicySource(*utiltesting.MakeMLPolicySourceWrapper().
							TorchPolicy().
							Obj(),
						).
						Obj(),
				),
				runtime.WithPodSet(constants.Node, ptr.To(constants.AncestorTrainer), 1, corev1.PodSpec{}, corev1ac.PodSpec().
					WithContainers(corev1ac.Container().WithName(constants.Node)),
				),
			),
			wantInfo: &runtime.Info{
				Labels:      make(map[string]string),
				Annotations: make(map[string]string),
				RuntimePolicy: runtime.RuntimePolicy{
					MLPolicySource: utiltesting.MakeMLPolicySourceWrapper().
						TorchPolicy().
						Obj(),
				},
				TemplateSpec: runtime.TemplateSpec{
					PodSets: []runtime.PodSet{{
						Name:              constants.Node,
						Ancestor:          ptr.To(constants.AncestorTrainer),
						Count:             ptr.To[int32](2),
						SinglePodRequests: make(corev1.ResourceList),
						Containers: []runtime.Container{{
							Name: constants.Node,
							Env: []corev1ac.EnvVarApplyConfiguration{
								{
									Name:  ptr.To(constants.TorchEnvNumNodes),
									Value: ptr.To("2"),
								},
								{
									Name:  ptr.To(constants.TorchEnvNumProcPerNode),
									Value: ptr.To("auto"),
								},
								{
									Name: ptr.To(constants.TorchEnvNodeRank),
									ValueFrom: &corev1ac.EnvVarSourceApplyConfiguration{
										FieldRef: &corev1ac.ObjectFieldSelectorApplyConfiguration{
											FieldPath: ptr.To(constants.JobCompletionIndexFieldPath),
										},
									},
								},
								{
									Name:  ptr.To(constants.TorchEnvMasterAddr),
									Value: ptr.To("lf-multi-node-0-0.lf-multi"),
								},
								{
									Name:  ptr.To(constants.TorchEnvMasterPort),
									Value: ptr.To(fmt.Sprintf("%d", constants.ContainerTrainerPort)),
								},
								{
									Name:  ptr.To(constants.LlamaFactoryEnvNumNodes),
									Value: ptr.To("2"),
								},
								{
									Name:  ptr.To(constants.LlamaFactoryEnvNumProcPerNode),
									Value: ptr.To("auto"),
								},
								{
									Name: ptr.To(constants.LlamaFactoryEnvNodeRank),
									ValueFrom: &corev1ac.EnvVarSourceApplyConfiguration{
										FieldRef: &corev1ac.ObjectFieldSelectorApplyConfiguration{
											FieldPath: ptr.To(constants.JobCompletionIndexFieldPath),
										},
									},
								},
								{
									Name:  ptr.To(constants.LlamaFactoryEnvMasterAddr),
									Value: ptr.To("lf-multi-node-0-0.lf-multi"),
								},
								{
									Name:  ptr.To(constants.LlamaFactoryEnvMasterPort),
									Value: ptr.To(fmt.Sprintf("%d", constants.ContainerTrainerPort)),
								},
								{
									Name:  ptr.To(constants.LlamaFactoryEnvForceTorchrun),
									Value: ptr.To("1"),
								},
							},
							Ports: []corev1ac.ContainerPortApplyConfiguration{{
								ContainerPort: ptr.To[int32](constants.ContainerTrainerPort),
							}},
						}},
					}},
				},
				Scheduler: &runtime.Scheduler{PodLabels: make(map[string]string)},
			},
		},
		"llamafactory without explicit command in TrainJob (command from runtime)": {
			trainJob: utiltesting.MakeTrainJobWrapper(metav1.NamespaceDefault, "lf-job").
				Trainer(
					utiltesting.MakeTrainJobTrainerWrapper().
						NumNodes(1).
						Container(
							"ghcr.io/kubeflow/trainer/llamafactory-trainer",
							nil,
							nil,
							corev1.ResourceList{
								"nvidia.com/gpu": resource.MustParse("1"),
							},
						).
						Obj(),
				).
				RuntimeRef(
					trainer.SchemeGroupVersion.WithKind(trainer.ClusterTrainingRuntimeKind),
					"llamafactory-lora-sft",
				).
				Obj(),
			info: runtime.NewInfo(
				runtime.WithMLPolicySource(
					utiltesting.MakeMLPolicyWrapper().
						WithMLPolicySource(*utiltesting.MakeMLPolicySourceWrapper().
							TorchPolicy().
							Obj(),
						).
						Obj(),
				),
				runtime.WithPodSet(constants.Node, ptr.To(constants.AncestorTrainer), 1, corev1.PodSpec{}, corev1ac.PodSpec().
					WithContainers(corev1ac.Container().
						WithName(constants.Node).
						WithCommand("llamafactory-cli", "train", "/config/training.yaml")),
				),
			),
			wantInfo: &runtime.Info{
				Labels:      make(map[string]string),
				Annotations: make(map[string]string),
				RuntimePolicy: runtime.RuntimePolicy{
					MLPolicySource: utiltesting.MakeMLPolicySourceWrapper().
						TorchPolicy().
						Obj(),
				},
				TemplateSpec: runtime.TemplateSpec{
					PodSets: []runtime.PodSet{{
						Name:              constants.Node,
						Ancestor:          ptr.To(constants.AncestorTrainer),
						Count:             ptr.To[int32](1),
						SinglePodRequests: make(corev1.ResourceList),
						Containers: []runtime.Container{{
							Name:    constants.Node,
							Command: []string{"llamafactory-cli", "train", "/config/training.yaml"},
							Env: []corev1ac.EnvVarApplyConfiguration{
								{
									Name:  ptr.To(constants.TorchEnvNumNodes),
									Value: ptr.To("1"),
								},
								{
									Name:  ptr.To(constants.TorchEnvNumProcPerNode),
									Value: ptr.To("auto"),
								},
								{
									Name: ptr.To(constants.TorchEnvNodeRank),
									ValueFrom: &corev1ac.EnvVarSourceApplyConfiguration{
										FieldRef: &corev1ac.ObjectFieldSelectorApplyConfiguration{
											FieldPath: ptr.To(constants.JobCompletionIndexFieldPath),
										},
									},
								},
								{
									Name:  ptr.To(constants.TorchEnvMasterAddr),
									Value: ptr.To("lf-job-node-0-0.lf-job"),
								},
								{
									Name:  ptr.To(constants.TorchEnvMasterPort),
									Value: ptr.To(fmt.Sprintf("%d", constants.ContainerTrainerPort)),
								},
								{
									Name:  ptr.To(constants.LlamaFactoryEnvNumNodes),
									Value: ptr.To("1"),
								},
								{
									Name:  ptr.To(constants.LlamaFactoryEnvNumProcPerNode),
									Value: ptr.To("auto"),
								},
								{
									Name: ptr.To(constants.LlamaFactoryEnvNodeRank),
									ValueFrom: &corev1ac.EnvVarSourceApplyConfiguration{
										FieldRef: &corev1ac.ObjectFieldSelectorApplyConfiguration{
											FieldPath: ptr.To(constants.JobCompletionIndexFieldPath),
										},
									},
								},
								{
									Name:  ptr.To(constants.LlamaFactoryEnvMasterAddr),
									Value: ptr.To("lf-job-node-0-0.lf-job"),
								},
								{
									Name:  ptr.To(constants.LlamaFactoryEnvMasterPort),
									Value: ptr.To(fmt.Sprintf("%d", constants.ContainerTrainerPort)),
								},
								{
									Name:  ptr.To(constants.LlamaFactoryEnvForceTorchrun),
									Value: ptr.To("1"),
								},
							},
							Ports: []corev1ac.ContainerPortApplyConfiguration{{
								ContainerPort: ptr.To[int32](constants.ContainerTrainerPort),
							}},
						}},
					}},
				},
				Scheduler: &runtime.Scheduler{PodLabels: make(map[string]string)},
			},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			torch, _ := New(ctx, nil, nil, nil)
			err := torch.(framework.EnforceMLPolicyPlugin).EnforceMLPolicy(tc.info, tc.trainJob)
			if diff := cmp.Diff(tc.wantMLPolicyError, err, cmpopts.EquateErrors()); len(diff) != 0 {
				t.Errorf("Unexpected error (-want,+got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantInfo, tc.info, cmpopts.EquateEmpty()); len(diff) != 0 {
				t.Errorf("Unexpected info (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestLlamaFactoryValidate(t *testing.T) {
	cases := map[string]struct {
		info         *runtime.Info
		oldObj       *trainer.TrainJob
		newObj       *trainer.TrainJob
		wantError    field.ErrorList
		wantWarnings admission.Warnings
	}{
		"llamafactory reserved env NNODES rejected": {
			info: runtime.NewInfo(
				runtime.WithMLPolicySource(utiltesting.MakeMLPolicyWrapper().
					WithMLPolicySource(*utiltesting.MakeMLPolicySourceWrapper().
						TorchPolicy().
						Obj(),
					).
					Obj(),
				),
			),
			newObj: utiltesting.MakeTrainJobWrapper(metav1.NamespaceDefault, "test").
				Trainer(utiltesting.MakeTrainJobTrainerWrapper().
					Container(
						"ghcr.io/kubeflow/trainer/llamafactory-trainer",
						[]string{"llamafactory-cli", "train"},
						[]string{"/config/training.yaml"},
						nil,
					).
					Env(
						corev1.EnvVar{
							Name:  constants.LlamaFactoryEnvNumNodes,
							Value: "2",
						},
					).
					Obj(),
				).
				Obj(),
			wantError: field.ErrorList{
				field.Invalid(
					field.NewPath("spec").Child("trainer").Child("env"),
					[]corev1.EnvVar{
						{
							Name:  constants.LlamaFactoryEnvNumNodes,
							Value: "2",
						},
					},
					fmt.Sprintf("must not have reserved envs, invalid envs configured: %v", func() []string {
						torchEnvs := sets.New[string]()
						torchEnvs.Insert(constants.LlamaFactoryEnvNumNodes)
						return sets.List(torchEnvs)
					}()),
				),
			},
		},
		"llamafactory reserved env FORCE_TORCHRUN rejected": {
			info: runtime.NewInfo(
				runtime.WithMLPolicySource(utiltesting.MakeMLPolicyWrapper().
					WithMLPolicySource(*utiltesting.MakeMLPolicySourceWrapper().
						TorchPolicy().
						Obj(),
					).
					Obj(),
				),
			),
			newObj: utiltesting.MakeTrainJobWrapper(metav1.NamespaceDefault, "test").
				Trainer(utiltesting.MakeTrainJobTrainerWrapper().
					Container(
						"ghcr.io/kubeflow/trainer/llamafactory-trainer",
						[]string{"llamafactory-cli", "train"},
						[]string{"/config/training.yaml"},
						nil,
					).
					Env(
						corev1.EnvVar{
							Name:  constants.LlamaFactoryEnvForceTorchrun,
							Value: "1",
						},
					).
					Obj(),
				).
				Obj(),
			wantError: field.ErrorList{
				field.Invalid(
					field.NewPath("spec").Child("trainer").Child("env"),
					[]corev1.EnvVar{
						{
							Name:  constants.LlamaFactoryEnvForceTorchrun,
							Value: "1",
						},
					},
					fmt.Sprintf("must not have reserved envs, invalid envs configured: %v", func() []string {
						torchEnvs := sets.New[string]()
						torchEnvs.Insert(constants.LlamaFactoryEnvForceTorchrun)
						return sets.List(torchEnvs)
					}()),
				),
			},
		},
		"non-llamafactory command does not reject llamafactory bridge envs": {
			info: runtime.NewInfo(
				runtime.WithMLPolicySource(utiltesting.MakeMLPolicyWrapper().
					WithMLPolicySource(*utiltesting.MakeMLPolicySourceWrapper().
						TorchPolicy().
						Obj(),
					).
					Obj(),
				),
			),
			newObj: utiltesting.MakeTrainJobWrapper(metav1.NamespaceDefault, "test").
				Trainer(utiltesting.MakeTrainJobTrainerWrapper().
					Env(
						corev1.EnvVar{
							Name:  constants.LlamaFactoryEnvNumNodes,
							Value: "2",
						},
					).
					Obj(),
				).
				Obj(),
		},
		"llamafactory reserved env rejected when command from runtime container": {
			info: runtime.NewInfo(
				runtime.WithMLPolicySource(utiltesting.MakeMLPolicyWrapper().
					WithMLPolicySource(*utiltesting.MakeMLPolicySourceWrapper().
						TorchPolicy().
						Obj(),
					).
					Obj(),
				),
				runtime.WithPodSet(constants.Node, ptr.To(constants.AncestorTrainer), 1, corev1.PodSpec{}, corev1ac.PodSpec().
					WithContainers(corev1ac.Container().
						WithName(constants.Node).
						WithCommand("llamafactory-cli", "train", "/config/training.yaml")),
				),
			),
			newObj: utiltesting.MakeTrainJobWrapper(metav1.NamespaceDefault, "test").
				Trainer(utiltesting.MakeTrainJobTrainerWrapper().
					Env(
						corev1.EnvVar{
							Name:  constants.LlamaFactoryEnvForceTorchrun,
							Value: "1",
						},
					).
					Obj(),
				).
				Obj(),
			wantError: field.ErrorList{
				field.Invalid(
					field.NewPath("spec").Child("trainer").Child("env"),
					[]corev1.EnvVar{
						{
							Name:  constants.LlamaFactoryEnvForceTorchrun,
							Value: "1",
						},
					},
					fmt.Sprintf("must not have reserved envs, invalid envs configured: %v", func() []string {
						torchEnvs := sets.New[string]()
						torchEnvs.Insert(constants.LlamaFactoryEnvForceTorchrun)
						return sets.List(torchEnvs)
					}()),
				),
			},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			torch, _ := New(ctx, nil, nil, nil)
			gotWarnings, gotError := torch.(framework.CustomValidationPlugin).Validate(ctx, tc.info, tc.oldObj, tc.newObj)
			if diff := cmp.Diff(tc.wantWarnings, gotWarnings, cmpopts.EquateEmpty()); len(diff) != 0 {
				t.Errorf("Unexpected warnings (-want,+got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantError, gotError, cmpopts.IgnoreFields(field.Error{}, "Detail")); len(diff) != 0 {
				t.Errorf("Unexpected error (-want,+got):\n%s", diff)
			}
		})
	}
}

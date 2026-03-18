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

	corev1ac "k8s.io/client-go/applyconfigurations/core/v1"
	"k8s.io/utils/ptr"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	"github.com/kubeflow/trainer/v2/pkg/apply"
	"github.com/kubeflow/trainer/v2/pkg/constants"
	"github.com/kubeflow/trainer/v2/pkg/runtime"
)

// enforceLlamaFactoryPolicy injects environment variables required by LLaMA Factory's launcher.
// LLaMA Factory reads unprefixed variables (NNODES, MASTER_ADDR, etc.) instead of the PET_
// prefixed variables used by torchrun. This function bridges the PET_ variables to their
// unprefixed equivalents and sets FORCE_TORCHRUN=1 to ensure torchrun is used for distributed training.
func enforceLlamaFactoryPolicy(
	trainerContainer *runtime.Container,
	trainJob *trainer.TrainJob,
	trainerPS *runtime.PodSet,
	numProcPerNode string,
) {
	numNodes := ptr.Deref(ptr.Deref(trainerPS, runtime.PodSet{}).Count, 1)
	masterAddr := fmt.Sprintf("%s-%s-0-0.%s", trainJob.Name, constants.Node, trainJob.Name)
	masterPort := fmt.Sprintf("%d", constants.ContainerTrainerPort)

	// Inject PET_MASTER_ADDR and PET_MASTER_PORT (same as torchrun path).
	apply.UpsertEnvVars(&trainerContainer.Env,
		*corev1ac.EnvVar().
			WithName(constants.TorchEnvMasterAddr).
			WithValue(masterAddr),
		*corev1ac.EnvVar().
			WithName(constants.TorchEnvMasterPort).
			WithValue(masterPort),
	)

	// Bridge PET_ variables to LLaMA Factory expected unprefixed names.
	apply.UpsertEnvVars(&trainerContainer.Env,
		*corev1ac.EnvVar().
			WithName(constants.LlamaFactoryEnvNumNodes).
			WithValue(fmt.Sprintf("%d", numNodes)),
		*corev1ac.EnvVar().
			WithName(constants.LlamaFactoryEnvNumProcPerNode).
			WithValue(numProcPerNode),
		*corev1ac.EnvVar().
			WithName(constants.LlamaFactoryEnvNodeRank).
			WithValueFrom(corev1ac.EnvVarSource().
				WithFieldRef(corev1ac.ObjectFieldSelector().
					WithFieldPath(constants.JobCompletionIndexFieldPath))),
		*corev1ac.EnvVar().
			WithName(constants.LlamaFactoryEnvMasterAddr).
			WithValue(masterAddr),
		*corev1ac.EnvVar().
			WithName(constants.LlamaFactoryEnvMasterPort).
			WithValue(masterPort),
		*corev1ac.EnvVar().
			WithName(constants.LlamaFactoryEnvForceTorchrun).
			WithValue("1"),
	)
}

/*
Copyright 2025.

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

package controller

import (
	"context"

	mlopsv1 "github.com/sathvik-8bit/model-operator/api/v1"
	batchv1 "k8s.io/api/batch/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// ModelDeploymentReconciler reconciles a ModelDeployment object
type ModelDeploymentReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=mlops.sathvik.dev,resources=modeldeployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=mlops.sathvik.dev,resources=modeldeployments/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=mlops.sathvik.dev,resources=modeldeployments/finalizers,verbs=update
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;update;patch;delete

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.

//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.21.0/pkg/reconcile
func (r *ModelDeploymentReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.FromContext(ctx)

	var md mlopsv1.ModelDeployment
	if err := r.Get(ctx, req.NamespacedName, &md); err != nil {
		// Object deleted — ignore
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	switch md.Status.Phase {
	case "":
		log.Info("New deployment detected, setting status to Validating")
		md.Status.Phase = "Validating"
		md.Status.Message = "Starting model validation"
		if err := r.Status().Update(ctx, &md); err != nil {
			log.Error(err, "unable to update status to Validating")
			return ctrl.Result{}, err
		}
		// Continue on next reconcile loop
		return ctrl.Result{Requeue: true}, nil

	case "Validating":
		// Check if validation Job exists. If not, create it.
		// If Job succeeded → proceed to deployment.
		// If Job failed → mark status as Failed.
		// (Implemented in next step.)

	default:
		log.Info("No action required", "phase", md.Status.Phase)
	}

	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *ModelDeploymentReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&mlopsv1.ModelDeployment{}).
		Owns(&batchv1.Job{}).
		Named("modeldeployment").
		Complete(r)
}

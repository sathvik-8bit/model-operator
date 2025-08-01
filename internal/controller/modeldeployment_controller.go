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
	"fmt"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	mlopsv1 "github.com/sathvik-8bit/model-operator/api/v1"
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/pointer"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/metrics"
)

var (
	deploymentsCreated = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "model_deployments_created_total",
			Help: "Total number of ModelDeployment resources created",
		},
	)

	validationsRun = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "model_validations_total",
			Help: "Number of model validation jobs run",
		},
		[]string{"status"}, // success / failure
	)
)

func init() {
	metrics.Registry.MustRegister(deploymentsCreated, validationsRun)
}

// ModelDeploymentReconciler reconciles a ModelDeployment object
type ModelDeploymentReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=mlops.sathvik.dev,resources=modeldeployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=mlops.sathvik.dev,resources=modeldeployments/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=mlops.sathvik.dev,resources=modeldeployments/finalizers,verbs=update
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=services,verbs=get;list;watch;create;update;patch;delete

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.

//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.21.0/pkg/reconcile
func (r *ModelDeploymentReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.FromContext(ctx)

	var md mlopsv1.ModelDeployment
	if md.ObjectMeta.DeletionTimestamp.IsZero() {
		// Not being deleted
		if !containsString(md.ObjectMeta.Finalizers, modelDeploymentFinalizer) {
			md.ObjectMeta.Finalizers = append(md.ObjectMeta.Finalizers, modelDeploymentFinalizer)
			if err := r.Update(ctx, &md); err != nil {
				return ctrl.Result{}, err
			}
			return ctrl.Result{}, nil // Finalizer added, requeue
		}
	} else {
		// Being deleted
		if containsString(md.ObjectMeta.Finalizers, modelDeploymentFinalizer) {
			log.Info("Cleaning up resources for deleted model", "name", md.Name)

			// Cleanup logic here
			if err := r.cleanupResources(ctx, &md); err != nil {
				return ctrl.Result{}, err
			}

			// Remove finalizer
			md.ObjectMeta.Finalizers = removeString(md.ObjectMeta.Finalizers, modelDeploymentFinalizer)
			if err := r.Update(ctx, &md); err != nil {
				return ctrl.Result{}, err
			}
			log.Info("Finalizer removed and cleanup complete")
			return ctrl.Result{}, nil
		}

		// Nothing to do — finalizer already gone
		return ctrl.Result{}, nil
	}

	
	if err := r.Get(ctx, req.NamespacedName, &md); err != nil {
		// Object deleted — ignore
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	switch md.Status.Phase {
	case "":
		log.Info("New deployment detected, setting status to Validating")
		deploymentsCreated.Inc()
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
		var jobList batchv1.JobList
		if err := r.List(ctx, &jobList, client.InNamespace(md.Namespace),
			client.MatchingLabels{"model": md.Name, "type": "validation"}); err != nil {
			return ctrl.Result{}, err
		}

		if len(jobList.Items) == 0 {
			job := buildValidationJob(&md)
			if err := ctrl.SetControllerReference(&md, job, r.Scheme); err != nil {
				return ctrl.Result{}, err
			}
			if err := r.Create(ctx, job); err != nil {
				return ctrl.Result{}, err
			}
			log.Info("Validation job created", "job", job.Name)
			validationsRun.WithLabelValues("started").Inc()			
			return ctrl.Result{RequeueAfter: 10 * time.Second}, nil
		}

		job := jobList.Items[0] // We only create one job per CR, so first match is enough

		for _, c := range job.Status.Conditions {
			if c.Type == batchv1.JobComplete && c.Status == corev1.ConditionTrue {
				// ✅ Validation succeeded
				validationsRun.WithLabelValues("success").Inc()
				md.Status.Phase = "Deploying"
				md.Status.Message = "Validation passed, proceeding to deploy model"
				if err := r.Status().Update(ctx, &md); err != nil {
					return ctrl.Result{}, err
				}
				return ctrl.Result{Requeue: true}, nil
			}

			if c.Type == batchv1.JobFailed && c.Status == corev1.ConditionTrue {
				// ❌ Validation failed
				validationsRun.WithLabelValues("failure").Inc()
				md.Status.Phase = "Failed"
				md.Status.Message = fmt.Sprintf("Validation failed: %s", c.Message)
				if err := r.Status().Update(ctx, &md); err != nil {
					return ctrl.Result{}, err
				}
				return ctrl.Result{}, nil // Stop processing
			}
		}
		
		case "Deploying":
			deploy := buildModelDeployment(&md)
			svc := buildModelService(&md)

			if err := ctrl.SetControllerReference(&md, deploy, r.Scheme); err != nil {
				return ctrl.Result{}, err
			}
			if err := ctrl.SetControllerReference(&md, svc, r.Scheme); err != nil {
				return ctrl.Result{}, err
			}

			if err := r.Create(ctx, deploy); err != nil && !errors.IsAlreadyExists(err) {
				return ctrl.Result{}, err
			}
			if err := r.Create(ctx, svc); err != nil && !errors.IsAlreadyExists(err) {
				return ctrl.Result{}, err
			}

			md.Status.Phase = "Ready"
			md.Status.Message = "Model deployed and service available"
			if err := r.Status().Update(ctx, &md); err != nil {
				return ctrl.Result{}, err
			}
			return ctrl.Result{}, nil

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

func buildValidationJob(md *mlopsv1.ModelDeployment) *batchv1.Job {
	backoff := int32(1)
	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("validate-%s", md.Name),
			Namespace: md.Namespace,
			Labels: map[string]string{
				"model": md.Name,
				"type":  "validation",
			},
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: &backoff,
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					RestartPolicy: corev1.RestartPolicyNever,
					Containers: []corev1.Container{
						{
							Name:  "validate",
							Image: "sathvik-8bit/validator:latest", // you’ll build this image
							Command: []string{"python", md.Spec.ValidateScript},
							Env: []corev1.EnvVar{
								{Name: "MODEL_URI", Value: md.Spec.ModelURI},
							},
							Resources: md.Spec.Resources,
						},
					},
				},
			},
		},
	}
}

func buildModelDeployment(md *mlopsv1.ModelDeployment) *appsv1.Deployment {
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("inference-%s", md.Name),
			Namespace: md.Namespace,
			Labels: map[string]string{
				"model": md.Name,
				"type":  "inference",
			},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: pointer.Int32(1),
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"model": md.Name},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"model": md.Name},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "inference",
							Image: runtimeToImage(md.Spec.Runtime),
							Env: []corev1.EnvVar{
								{Name: "MODEL_URI", Value: md.Spec.ModelURI},
							},
							Resources: md.Spec.Resources,
							Ports: []corev1.ContainerPort{
								{ContainerPort: 8000},
							},
						},
					},
				},
			},
		},
	}
}


func buildModelService(md *mlopsv1.ModelDeployment) *corev1.Service {
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("svc-%s", md.Name),
			Namespace: md.Namespace,
		},
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{"model": md.Name},
			Ports: []corev1.ServicePort{
				{Port: 80, TargetPort: intstr.FromInt(8000)},
			},
			Type: corev1.ServiceTypeClusterIP,
		},
	}
}

func runtimeToImage(runtime string) string {
	switch runtime {
	case "triton":
		return "nvcr.io/nvidia/tritonserver:22.08-py3"
	case "torchserve":
		return "pytorch/torchserve:latest"
	default:
		return "unknown"
	}
}

const modelDeploymentFinalizer = "mlops.sathvik.dev/finalizer"

// Helper functions for string slice operations
func containsString(slice []string, str string) bool {
	for _, item := range slice {
		if item == str {
			return true
		}
	}
	return false
}

func removeString(slice []string, str string) []string {
	result := []string{}
	for _, item := range slice {
		if item != str {
			result = append(result, item)
		}
	}
	return result
}

// cleanupResources handles cleanup when a ModelDeployment is deleted
func (r *ModelDeploymentReconciler) cleanupResources(ctx context.Context, md *mlopsv1.ModelDeployment) error {
	labels := client.MatchingLabels{"model": md.Name}

	var jobs batchv1.JobList
	if err := r.List(ctx, &jobs, client.InNamespace(md.Namespace), labels); err != nil {
		return err
	}
	for _, job := range jobs.Items {
		if err := r.Delete(ctx, &job); err != nil {
			return err
		}
	}

	var deploys appsv1.DeploymentList
	if err := r.List(ctx, &deploys, client.InNamespace(md.Namespace), labels); err != nil {
		return err
	}
	for _, d := range deploys.Items {
		if err := r.Delete(ctx, &d); err != nil {
			return err
		}
	}

	var svcs corev1.ServiceList
	if err := r.List(ctx, &svcs, client.InNamespace(md.Namespace), labels); err != nil {
		return err
	}
	for _, s := range svcs.Items {
		if err := r.Delete(ctx, &s); err != nil {
			return err
		}
	}

	return nil
}

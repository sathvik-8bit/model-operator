package v1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ModelDeploymentSpec defines the desired state of ModelDeployment
type ModelDeploymentSpec struct {
	
	ModelURI string `json:"modelURI"`

	// +kubebuilder:validation:Enum=triton;torchserve
	Runtime string `json:"runtime"`

	Resources corev1.ResourceRequirements `json:"resources"`

	// Optional script to run for validation
	ValidateScript string `json:"validateScript,omitempty"`

	Autoscale bool `json:"autoscale"`
	Version string `json:"version"`
}

// ModelDeploymentStatus defines the observed state of ModelDeployment
type ModelDeploymentStatus struct {
	// +kubebuilder:validation:Enum=Pending;Validating;Deploying;Ready;Failed
	Phase string `json:"phase,omitempty"`

	Message string `json:"message,omitempty"`
	Version string `json:"version,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// ModelDeployment is the Schema for the modeldeployments API
type ModelDeployment struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   ModelDeploymentSpec   `json:"spec"`
	Status ModelDeploymentStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// ModelDeploymentList contains a list of ModelDeployment
type ModelDeploymentList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []ModelDeployment `json:"items"`
}

func init() {
	SchemeBuilder.Register(&ModelDeployment{}, &ModelDeploymentList{})
}

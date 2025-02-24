using MediatR;

namespace MetricsSubscriber.Models.Vertical;

public readonly record struct ModelConfigurationNotification(
	string Name,
	IReadOnlyList<int> Layers,
	double LearningRate
) : INotification;

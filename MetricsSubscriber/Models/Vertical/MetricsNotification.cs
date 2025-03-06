using MediatR;

namespace MetricsSubscriber.Models.Vertical;

public readonly record struct MetricsNotification(
	int Epoch,
	double Loss,
	double Accuracy,
	double MeanPPV,
	double MeanFPR,
	double MeanRecall
) : INotification;

using MediatR;

namespace MetricsSubscriber.Models.Vertical;

public record MetricsUpdatedEvent(double Epoch, double Loss, double Accuracy, double PPV, double Recall, double FPR) : INotification;


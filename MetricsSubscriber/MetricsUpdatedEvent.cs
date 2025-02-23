using MediatR;

namespace MetricsSubscriber;

public record MetricsUpdatedEvent(double Epoch, double Loss, double Accuracy, double PPV, double Recall, double FPR) : INotification;


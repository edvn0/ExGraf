using MediatR;
using MetricsSubscriber.Models.Vertical;
using Microsoft.Extensions.Logging;

namespace MetricsSubscriber;

public class MetricsNotificationHandler(ILogger<MetricsNotificationHandler> logger) : INotificationHandler<MetricsNotification>
{
	public Task Handle(MetricsNotification notification, CancellationToken cancellationToken)
	{
		logger.LogInformation(
			"Epoch {Epoch}: Loss={Loss:.4f}, Accuracy={Accuracy:.2f}%, PPV={PPV:.4f}, FPR={FPR:.4f}, Recall={Recall:.4f}",
			notification.Epoch,
			notification.Loss,
			notification.Accuracy * 100.0,
			notification.MeanPPV,
			notification.MeanFPR,
			notification.MeanRecall);

		return Task.CompletedTask;
	}
}


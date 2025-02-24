using MediatR;
using MetricsSubscriber.Models.Vertical;
using Microsoft.Extensions.Logging;
using ScottPlot;

namespace MetricsSubscriber;

public class MetricsUpdatedEventHandler : INotificationHandler<MetricsUpdatedEvent>
{
	private readonly List<double> _epochs = [];
	private readonly List<double> _loss = [];
	private readonly List<double> _accuracy = [];
	private readonly List<double> _ppv = [];
	private readonly List<double> _recall = [];
	private readonly List<double> _fpr = [];
	private readonly Lock _lock = new();
	private readonly string _outputFile = "metrics.png";

	public Task Handle(MetricsUpdatedEvent notification, CancellationToken cancellationToken)
	{
		lock (_lock)
		{
			_epochs.Add(notification.Epoch);
			_loss.Add(notification.Loss);
			_accuracy.Add(notification.Accuracy);
			_ppv.Add(notification.PPV);
			_recall.Add(notification.Recall);
			_fpr.Add(notification.FPR);
		}

		SavePlot();
		return Task.CompletedTask;
	}

	private void SavePlot()
	{
		var plt = new Plot();

		{
			_lock.EnterScope();
			if (_epochs.Count == 0) return;

			plt.Title("Training Metrics");
			plt.XLabel("Epoch");

			plt.Add.Scatter(_epochs.ToArray(), _loss.ToArray(), ScottPlot.Color.FromColor(System.Drawing.Color.Red));
			plt.Add.Scatter(_epochs.ToArray(), _accuracy.ToArray(), ScottPlot.Color.FromColor(System.Drawing.Color.Green));
			plt.Add.Scatter(_epochs.ToArray(), _ppv.ToArray(), ScottPlot.Color.FromColor(System.Drawing.Color.Blue));
			plt.Add.Scatter(_epochs.ToArray(), _recall.ToArray(), ScottPlot.Color.FromColor(System.Drawing.Color.Orange));
			plt.Add.Scatter(_epochs.ToArray(), _fpr.ToArray(), ScottPlot.Color.FromColor(System.Drawing.Color.Purple));

			plt.ShowLegend();
		}

		plt.SaveSvg(_outputFile, 1000, 800);
		Console.WriteLine($"Updated plot saved as '{_outputFile}'");
	}
}

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


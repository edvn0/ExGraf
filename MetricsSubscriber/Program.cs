using System.Reflection;
using MediatR;
using MetricsSubscriber;
using MetricsSubscriber.Configuration;
using MetricsSubscriber.Implementations;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

using var cancellationTokenSource = new CancellationTokenSource();

Console.CancelKeyPress += (sender, e) =>
{
	e.Cancel = true;
	cancellationTokenSource.Cancel();
};



var host = Host.CreateDefaultBuilder()
	.ConfigureServices((context, services) =>
	{
		services.AddMediatR(cfg => cfg.RegisterServicesFromAssembly(Assembly.GetExecutingAssembly()));
		services.AddTransient<ISubscriberSocket, NetMqSubscriberSocket>();

		services.AddMessageHandlingFor<MetricsUpdatedEvent, MetricsMessageParser>();

		services.AddLogging();
	})
	.Build();

try
{
	await host.StartAsync(cancellationTokenSource.Token);
	await host.WaitForShutdownAsync(cancellationTokenSource.Token);
}
finally
{
	if (host is IAsyncDisposable disposableHost)
		await disposableHost.DisposeAsync();
	else
		host.Dispose();
}

